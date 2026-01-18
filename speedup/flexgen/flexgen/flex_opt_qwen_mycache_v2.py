"""
flex_opt.py (with sparse/selective KV cache + prefetch + partial cache)

Usage:
python3 -m flexllmgen.flex_opt \
  --model Qwen/Qwen3-8B \
  --path /path/to/hf_dir \
  --gpu-batch-size 1 --num-gpu-batches 1 \
  --prompt-len 2048 --gen-len 256 \
  --warmup-input-path warmup.txt \
  --test-input-path test.txt
"""

import argparse
import dataclasses
import os
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config
from flexgen.pytorch_backend import (
    TorchDevice, TorchDisk, TorchMixedDevice, DeviceType,
    general_copy, fix_recursive_import, TorchTensor
)
from flexgen.timer import timers
from flexgen.utils import (
    Task, ExecutionEnv, GB, ValueHolder,
    array_1d, array_2d, array_3d, str2bool,
    project_decode_latency, torch_dtype_to_np_dtype,
    write_benchmark_log
)

from infinigen.partial_weight_generation_controller import (
    set_partial_cache_gqa, set_partial_weight
)
from flexgen.cache_selection_controller_v2 import CacheManager

from safetensors.torch import load_file

fix_recursive_import()
DUMMY_WEIGHT = "_DUMMY_"


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    input_dtype = input.dtype
    variance = input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden_states = input * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    overlap: bool
    sep_layer: bool
    pin_weight: bool
    cpu_cache_compute: bool
    attn_sparsity: float

    compress_weight: bool
    comp_weight_config: CompressionConfig

    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = env.gpu
        self.task = None

    def set_task(self, task): self.task = task

    def init_weight(self, weight_home, state_dict=None, layer_id=None):
        if state_dict is None:
            raise ValueError("state_dict required")
        device = self.env.gpu.dev
        dtype = self.config.torch_dtype
        embed_tokens_weight = state_dict["model.embed_tokens.weight"].to(device, dtype)
        weight_home.val = {"embed_tokens_weight": embed_tokens_weight}

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        x = hidden.val
        if hasattr(x, "data"):
            x = x.data
        input_ids = x.long()
        w_tok = weight_read_buf.val["embed_tokens_weight"]
        x = F.embedding(input_ids, w_tok, padding_idx=self.config.pad_token_id)
        hidden.val = x


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = env.gpu
        self.task = None

    def set_task(self, task): self.task = task

    def init_weight(self, weight_home, state_dict=None, layer_id=None):
        if state_dict is None:
            raise ValueError("state_dict required")
        device = self.env.gpu.dev
        dtype = self.config.torch_dtype
        if "lm_head.weight" in state_dict:
            lm_head_weight = state_dict["lm_head.weight"]
        else:
            lm_head_weight = state_dict["model.embed_tokens.weight"]
        weight_home.val = {"lm_head_weight": lm_head_weight.to(device, dtype)}

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        x = hidden.val
        if x.dim() == 3:
            x = x[:, -1, :]
        w = weight_read_buf.val["lm_head_weight"]
        logits = F.linear(x, w)
        if self.task.temperature != 1.0:
            logits = logits / self.task.temperature
        if self.task.do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(logits, dim=-1)
        hidden.val = next_token.unsqueeze(1)


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_id = layer_id
        self.compute = env.gpu
        self.task = None

    def set_task(self, task): self.task = task

    def init_weight(self, weight_home, state_dict=None, layer_id=None):
        if state_dict is None:
            raise ValueError("state_dict required")
        device = self.env.gpu.dev
        dtype = self.config.torch_dtype
        prefix = f"model.layers.{layer_id}.mlp."
        norm_prefix = f"model.layers.{layer_id}.post_attention_layernorm."
        weight_home.val = {
            "gate_proj_weight": state_dict[prefix + "gate_proj.weight"].to(device, dtype),
            "up_proj_weight": state_dict[prefix + "up_proj.weight"].to(device, dtype),
            "down_proj_weight": state_dict[prefix + "down_proj.weight"].to(device, dtype),
            "norm_weight": state_dict[norm_prefix + "weight"].to(device, dtype),
        }

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i, k):
        x = hidden.val
        w = weight_read_buf.val
        x = rms_norm(x, w["norm_weight"], eps=self.config.rms_norm_eps)
        gate = F.linear(x, w["gate_proj_weight"])
        up = F.linear(x, w["up_proj_weight"])
        x = gate * F.silu(up)
        x = F.linear(x, w["down_proj_weight"])
        hidden.val = x


class SelfAttention:
    """
    Sparse/selective cache + prefetch + partial cache
    Requires backend kernels:
      - TorchDevice.mha_qwen_infin_v2(...)
      - TorchDevice.mha_gen_qwen_infin_v2(...)
      - TorchDevice.init_cache_one_gpu_batch_infin(...)
    """
    def __init__(
        self, config, env, policy, layer_id, cache_manager,
        enable_prefetching: bool,
        partial_weight_ratio=0.2, alpha=4, max_num_kv=400
    ):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_id = layer_id

        self.compute = env.gpu
        self.attention_compute = env.gpu

        self.task = None
        self.enable_prefetching = enable_prefetching
        self.prefetch_idx = None

        self.partial_index = None
        self.alpha = alpha
        self.max_num_kv = max_num_kv
        self.partial_weight_ratio = partial_weight_ratio if layer_id > 1 else None

        self.prefetch_kv = None  # temp buffer

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        if not isinstance(cache_manager, CacheManager):
            raise ValueError("SelfAttention! Initial Error, please use cache manager as parameter!")
        self._cache_manager = cache_manager
        self._debug_logged = False

    def set_task(self, task): self.task = task

    def init_weight(self, weight_home, state_dict=None, layer_id=None):
        if state_dict is None:
            raise ValueError("state_dict required")
        device = self.env.gpu.dev
        dtype = self.config.torch_dtype
        prefix = f"model.layers.{layer_id}."
        weight_home.val = {
            "q_proj_weight": state_dict[prefix + "self_attn.q_proj.weight"].to(device, dtype),
            "k_proj_weight": state_dict[prefix + "self_attn.k_proj.weight"].to(device, dtype),
            "v_proj_weight": state_dict[prefix + "self_attn.v_proj.weight"].to(device, dtype),
            "o_proj_weight": state_dict[prefix + "self_attn.o_proj.weight"].to(device, dtype),
            "norm_weight": state_dict[prefix + "input_layernorm.weight"].to(device, dtype),
            "q_norm_weight": state_dict.get(prefix + "self_attn.q_norm.weight", None),
            "k_norm_weight": state_dict.get(prefix + "self_attn.k_norm.weight", None),
        }
        if weight_home.val["q_norm_weight"] is not None:
            weight_home.val["q_norm_weight"] = weight_home.val["q_norm_weight"].to(device, dtype)
        if weight_home.val["k_norm_weight"] is not None:
            weight_home.val["k_norm_weight"] = weight_home.val["k_norm_weight"].to(device, dtype)

    def init_cache_one_gpu_batch(self, cache_home: ValueHolder):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        k_cache, v_cache = device.init_cache_one_gpu_batch_infin(self.config, self.task, self.policy)
        cache_home.store((k_cache, v_cache))
        if self.layer_id == 0:  # 只打印一次避免刷屏，你也可以改成 layer_id > 1 等
            print(f"[cache init] device={device.device_type} layer={self.layer_id} "
                  f"k_cache.shape={getattr(k_cache, 'shape', None)} v_cache.shape={getattr(v_cache, 'shape', None)}",
                  flush=True)

        if self.layer_id > 1:
            self.prefetch_kv = None

    def load_cache(self, cache_home, cache_read_buf, i: int):
        if i == 0:
            return
        k_home, v_home = cache_home.val
        dst = self.attention_compute
        indices = (slice(0, self.task.prompt_len + i), slice(0, k_home.shape[1]))
        cache_read_buf.store((
            k_home.smart_copy(dst, indices),
            v_home.smart_copy(dst, indices),
        ))

    def prefetch_cache(self, cache_home, cache_read_buf, i: int, prefetch_idx: torch.Tensor, prefetch_cache_stream: torch.cuda.Stream):
        if i == 0:
            return
        if isinstance(prefetch_idx, tuple):
            real_prefetch_idx = prefetch_idx[0]
            pad_idx = prefetch_idx[1]
        else:
            real_prefetch_idx = prefetch_idx
            pad_idx = None

        k_home, v_home = cache_home.val

        if self.policy.compress_cache:
            path = 0
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0

        if path == 0:
            if self.policy.attn_sparsity >= 1.0:
                if not self._debug_logged:
                    print(
                        f"[prefetch] layer={self.layer_id} "
                        f"k_cache.shape={getattr(k_home.data, 'shape', None)} "
                        f"v_cache.shape={getattr(v_home.data, 'shape', None)} "
                        f"selected_idx.shape={getattr(real_prefetch_idx, 'shape', None)}",
                        flush=True
                    )
                    self._debug_logged = True
                prefetch_idx_int = real_prefetch_idx.cpu().int()
                if pad_idx is None:
                    pad_idx_int = torch.zeros((1, 1, prefetch_idx_int.shape[2]),
                                              dtype=torch.int32)
                else:
                    pad_idx_int = pad_idx.cpu().int()

                k_data = k_home.data
                v_data = v_home.data
                cache_dtype = k_home.dtype
                # if isinstance(k_data, torch.Tensor) and k_data.dtype == torch.bfloat16:
                #     k_data = k_data.to(torch.float16)
                #     v_data = v_data.to(torch.float16)
                #     cache_dtype = config.torch_dtype

                group_gpu_k, group_gpu_v, group_unhit = self._cache_manager.unified_load_api(
                    self.layer_id, prefetch_cache_stream, prefetch_idx_int, pad_idx_int,
                    k_data, v_data, cache_dtype
                )

                cache_read_buf.store(
                    ((group_gpu_k, True),
                     (group_gpu_v, True))
                )
        elif path == 1 or path == 2:
            raise ValueError(f"Not implemented path: {path}")
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i: int):
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:
            return

        if i == 0:
            indices = (slice(0, k_new.shape[0]), slice(0, k_new.shape[1]))
        else:
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos), slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def forward(
        self,
        hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf,
        i, k,
        warmup: bool,
        partial_weight_read_buf: ValueHolder,
        partial_cache_read_buf: ValueHolder,
        speculation_stream: torch.cuda.Stream,
        prev_partial_cache_read_buf: Optional[ValueHolder],
        prev_partial_weight_read_buf: Optional[ValueHolder],
        weight_home: ValueHolder
    ):
        x = hidden.val
        weights = weight_read_buf.val

        donate = [False] * 14

        w_q = weights["q_proj_weight"]
        w_k = weights["k_proj_weight"]
        w_v = weights["v_proj_weight"]
        w_out = weights["o_proj_weight"]
        w_ln = weights["norm_weight"]
        q_ln = weights["q_norm_weight"]
        k_ln = weights["k_norm_weight"]

        n_head = self.num_attention_heads
        head_dim = self.head_dim

        mask = attention_mask.val

        # For decode sparse kernel
        if self.enable_prefetching and (i > 0):
            p_w_q = partial_weight_read_buf.val

        if i == 0:
            # Prefill: build full KV, and build partial for prev layer buffers
            h_out, new_k_cache, new_v_cache, w_q_upd, w_k_upd, self.partial_index = self.compute.mha_qwen_infin_v2(
                x, mask, w_q, w_k, w_v, w_out, w_ln, q_ln, k_ln,
                n_head, donate,
                False, None,
                self.config.rms_norm_eps, self.head_dim,
                self.num_key_value_groups, self.num_key_value_heads,
                None,
                warmup, self.partial_weight_ratio
            )
            cache_write_buf.store((new_k_cache, new_v_cache))

            # store partial into "prev" buffers (pipeline design)
            if (prev_partial_cache_read_buf is not None) and (not warmup):
                prev_partial_cache_read_buf.store(
                    set_partial_cache_gqa(
                        new_k_cache.data, self.partial_index, n_head,
                        self.num_key_value_heads, head_dim
                    )
                )
                prev_partial_weight_read_buf.store(set_partial_weight(w_q_upd.data, self.partial_index, n_head, head_dim))

            # warmup may update weights
            if warmup:
                weights["q_proj_weight"] = w_q_upd
                weights["k_proj_weight"] = w_k_upd

            hidden.val = h_out.data
            return

        # Decode
        (k_cache_buf, _), (v_cache_buf, _) = cache_read_buf.pop()

        def _merge_group_cache(group_cache):
            if isinstance(group_cache, (list, tuple)):
                tensors = []
                for item in group_cache:
                    if isinstance(item, (list, tuple)):
                        tensors.extend(item)
                    else:
                        tensors.append(item)
                if len(tensors) == 1:
                    return tensors[0]
                tensors = [t.data if isinstance(t, TorchTensor) else t for t in tensors]
                return torch.cat(tensors, dim=0)
            return group_cache

        k_cache_buf = _merge_group_cache(k_cache_buf)
        v_cache_buf = _merge_group_cache(v_cache_buf)

        if not isinstance(k_cache_buf, TorchTensor):
            k_cache_buf = TorchTensor.create_from_torch(k_cache_buf, self.compute)
            v_cache_buf = TorchTensor.create_from_torch(v_cache_buf, self.compute)

        k_cache_buf.data = k_cache_buf.data.to(torch.bfloat16)
        v_cache_buf.data = v_cache_buf.data.to(torch.bfloat16)

        if self.enable_prefetching:
            partial_k_cache = partial_cache_read_buf.val
            h_out, new_k_cache, new_v_cache, self.prefetch_idx = self.compute.mha_gen_qwen_infin_v2(
                x, mask, w_q, w_k, w_v, w_out, w_ln, q_ln, k_ln,
                n_head,
                k_cache_buf, v_cache_buf,
                donate,
                self.policy.attn_sparsity, self.policy.compress_cache, self.policy.comp_cache_config,
                self.config.rms_norm_eps, self.head_dim,
                self.num_key_value_groups, self.num_key_value_heads,
                None,
                p_w_q, partial_k_cache, speculation_stream,
                self.alpha, self.max_num_kv,
                prefetch_grouped=True
            )
        else:
            # fallback (if you have a non-v2 kernel, adapt here)
            h_out, new_k_cache, new_v_cache, _ = self.compute.mha_gen_qwen_infin_v2(
                x, mask, w_q, w_k, w_v, w_out, w_ln, q_ln, k_ln,
                n_head,
                k_cache_buf, v_cache_buf,
                donate,
                self.policy.attn_sparsity, self.policy.compress_cache, self.policy.comp_cache_config,
                self.config.rms_norm_eps, self.head_dim,
                self.num_key_value_groups, self.num_key_value_heads,
                None,
                None, None, None,
                self.alpha, self.max_num_kv
            )

        cache_write_buf.store((new_k_cache, new_v_cache))

        if (prev_partial_cache_read_buf is not None) and (self.layer_id > 1):
            prev_partial_cache_read_buf.val = torch.cat(
                (prev_partial_cache_read_buf.val, set_partial_cache_gqa(
                    new_k_cache.data, self.partial_index, n_head,
                    self.num_key_value_heads, head_dim
                ))
            )

        hidden.val = h_out.data


class OptLM:
    def __init__(
        self,
        config: Union[str, OptConfig],
        env: ExecutionEnv,
        path: str,
        policy: Policy,
        partial_weight_ratio=0.2,
        alpha=4,
        max_num_kv=400,
        gpu_cache_num: int = 0,
        gpu_cache_pred: float = 1.0,
        cpu_cache_pred: float = 1.0,
    ):
        if isinstance(config, str):
            config = get_opt_config(config)

        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.gpu_cache_num = int(gpu_cache_num)
        self.gpu_cache_pred = float(gpu_cache_pred)
        self.cpu_cache_pred = float(cpu_cache_pred)

        self.head_num = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        cache_dtype = config.torch_dtype

        original_head_group_ids = {}
        for l in range(self.config.num_hidden_layers):
            original_head_group_ids[l] = [list(range(self.head_num))]
        self._cache_manager = CacheManager(
            basic_group_head_ids=original_head_group_ids,
            layer_head_num=self.head_num,
            batch_size=self.policy.gpu_batch_size,
            head_num=self.head_num,
            max_sparse_len=max_num_kv,
            head_dim=self.head_dim,
            dtype=cache_dtype,
            gpu_cache_pred=self.gpu_cache_pred,
            cpu_cache_pred=self.cpu_cache_pred,
        )

        # ---- load state_dict from safetensors ----
        print(f"Loading model from safetensors in: {path}")
        safetensor_files = sorted([f for f in os.listdir(path) if f.endswith(".safetensors")])
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")

        state_dict = {}
        for f in safetensor_files:
            file_path = os.path.join(path, f)
            print(f"Loading {file_path}...")
            tensors = load_file(file_path, device="cpu")
            state_dict.update(tensors)
        print(f"Loaded {len(state_dict)} tensors.")
        self.model_state_dict = state_dict

        # ---- build layers ----
        self.layers = []
        self.attn_layer = []  # indices of attention layers in self.layers

        self.layers.append(InputEmbed(self.config, self.env, self.policy))

        for i in range(self.config.num_hidden_layers):
            if self.policy.sep_layer:
                enable_prefetching = not (i == 0 or i == self.config.num_hidden_layers - 1)
                self.layers.append(SelfAttention(
                    self.config, self.env, self.policy, i, self._cache_manager,
                    enable_prefetching=enable_prefetching,
                    partial_weight_ratio=partial_weight_ratio,
                    alpha=alpha, max_num_kv=max_num_kv
                ))
                self.attn_layer.append(len(self.layers) - 1)
                self.layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                # If you truly need TransformerLayer, implement it. This file assumes sep_layer=True.
                raise NotImplementedError("This file assumes --sep-layer True for simplicity.")

        self.layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.num_layers = len(self.layers)

        # ---- streams for overlap/prefetch ----
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        self.speculation_stream = torch.cuda.Stream()
        self.prefetch_cache_stream = torch.cuda.Stream()
        self.prefetch_evt = torch.cuda.Event()

        # ---- buffers ----
        L = self.num_layers
        B = self.policy.num_gpu_batches
        self.cache_home = array_2d(L, B, ValueHolder)
        self.cache_read_buf = array_2d(L, B, ValueHolder)
        self.cache_write_buf = array_2d(L, B, ValueHolder)
        self.weight_home = array_1d(L, ValueHolder)
        self.weight_read_buf = array_1d(L, ValueHolder)
        self.attention_mask = array_1d(B, ValueHolder)

        # partial buffers
        self.partial_cache_read_buf = array_2d(L, B, ValueHolder)
        self.partial_weight_read_buf = array_1d(L, ValueHolder)

        self.task = None
        self.warmup = False
        self.execute_gen_len = None

        # ---- init weights (resident on GPU) ----
        self.init_all_weights_from_safetensors()

    def init_all_weights_from_safetensors(self):
        print("Initializing weights from state_dict...")
        sd = self.model_state_dict
        for j, layer in enumerate(self.layers):
            if isinstance(layer, (SelfAttention, MLP)):
                layer.init_weight(self.weight_home[j], state_dict=sd, layer_id=layer.layer_id)
            else:
                layer.init_weight(self.weight_home[j], state_dict=sd, layer_id=None)
        del self.model_state_dict
        torch.cuda.empty_cache()

    def set_task(self, task: Task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_cache(self, j, k):
        if isinstance(self.layers[j], SelfAttention):
            self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_weight(self, i, j, k):
        if k == 0:
            self.weight_read_buf[j].val = self.weight_home[j].val

    def load_cache(self, i, j, k, overlap=True):
        if i == 0:
            return
        if not isinstance(self.layers[j], SelfAttention):
            return

        # IMPORTANT: from 3rd attention layer onward, we rely on prefetch_cache to fill cache_read_buf
        if j in self.attn_layer[2:]:
            return

        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def prefetch_cache(self, i, attn_layer_idx_in_layers, k):
        if i == 0:
            return
        cur_attn = attn_layer_idx_in_layers
        if cur_attn not in self.attn_layer:
            return
        pos = self.attn_layer.index(cur_attn)
        if pos + 1 >= len(self.attn_layer):
            return
        next_attn = self.attn_layer[pos + 1]

        prefetch_idx = self.layers[cur_attn].prefetch_idx
        if prefetch_idx is None:
            return

        self.layers[next_attn].prefetch_cache(
            self.cache_home[next_attn][k],
            self.cache_read_buf[next_attn][k],
            i,
            prefetch_idx,
            self.prefetch_cache_stream
        )

    def store_cache(self, i, j, k, overlap=True):
        if not isinstance(self.layers[j], SelfAttention):
            return
        if i == self.task.gen_len - 1:
            self.cache_write_buf[j][k].pop()
            return
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        device = self.env.gpu.dev
        dtype = torch.long
        bs = self.policy.gpu_batch_size
        left, right = k * bs, (k + 1) * bs

        if j == 0:
            if i == 0:
                token_ids = self.output_ids[left:right, :self.task.prompt_len]
            else:
                pos = self.task.prompt_len + i - 1
                token_ids = self.output_ids[left:right, pos:pos + 1]
            self.hidden[i][j][k].val = torch.tensor(token_ids, device=device, dtype=dtype)
        else:
            self.hidden[i][j][k].val = self.hidden[i][j - 1][k].val

    def store_hidden(self, i, j, k):
        if j != self.num_layers - 1:
            return
        bs = self.policy.gpu_batch_size
        left, right = k * bs, (k + 1) * bs
        ids = self.hidden[i][j][k].val.detach().cpu().numpy()
        pos = self.task.prompt_len + i
        self.output_ids[left:right, pos:pos + 1] = ids

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            old = mask.val
            mask.val = torch.cat([old, torch.ones_like(old[:, :1])], dim=1)
            return

        bs = self.policy.gpu_batch_size
        left = k * bs
        right = left + bs
        input_ids = self.output_ids[left:right, :self.task.prompt_len]
        self.attention_mask[k].val = torch.tensor(
            input_ids != self.config.pad_token_id,
            dtype=torch.bool,
            device=self.env.gpu.dev
        )

    def compute_layer(self, i, j, k):
        layer = self.layers[j]
        warmup_state = (k == 0) and self.warmup

        if isinstance(layer, SelfAttention):
            if j in self.attn_layer[2:]:
                prev_attn = self.attn_layer[self.attn_layer.index(j) - 1]
                layer.forward(
                    self.hidden[i][j][k],
                    self.cache_read_buf[j][k],
                    self.weight_read_buf[j],
                    self.attention_mask[k],
                    self.cache_write_buf[j][k],
                    i=i, k=k,
                    warmup=warmup_state,
                    partial_weight_read_buf=self.partial_weight_read_buf[j],
                    partial_cache_read_buf=self.partial_cache_read_buf[j][k],
                    speculation_stream=self.speculation_stream,
                    prev_partial_cache_read_buf=self.partial_cache_read_buf[prev_attn][k],
                    prev_partial_weight_read_buf=self.partial_weight_read_buf[prev_attn],
                    weight_home=self.weight_home[j],
                )
            else:
                layer.forward(
                    self.hidden[i][j][k],
                    self.cache_read_buf[j][k],
                    self.weight_read_buf[j],
                    self.attention_mask[k],
                    self.cache_write_buf[j][k],
                    i=i, k=k,
                    warmup=warmup_state,
                    partial_weight_read_buf=self.partial_weight_read_buf[j],
                    partial_cache_read_buf=self.partial_cache_read_buf[j][k],
                    speculation_stream=self.speculation_stream,
                    prev_partial_cache_read_buf=None,
                    prev_partial_weight_read_buf=None,
                    weight_home=self.weight_home[j],
                )
            return

        # MLP / Embeds
        layer.forward(
            self.hidden[i][j][k],
            self.cache_read_buf[j][k],
            self.weight_read_buf[j],
            self.attention_mask[k],
            self.cache_write_buf[j][k],
            i=i, k=k
        )

    def sync(self):
        torch.cuda.synchronize()

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)

                    self.compute_layer(i, j, k)
                    self.sync()

                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)

                    # schedule prefetch after attention layer decoding
                    if (j in self.attn_layer[1:-1]) and (i > 0):
                        self.prefetch_cache(i, j, k)
                        self.prefetch_evt.record()

            timers("generate").stop()

    def generate(
        self,
        inputs: Union[np.ndarray, List[List[int]]],
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        stop: Optional[int] = None,
        debug_mode: Optional[str] = None,
        cut_gen_len: Optional[int] = None,
        verbose: int = 0,
        warmup: bool = False,
    ):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        self.set_task(task)
        self.warmup = warmup
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        if self.config.pad_token_id is None:
            self.config.pad_token_id = 0

        prompt_len, gen_len = task.prompt_len, task.gen_len
        num_prompts = len(task.inputs)
        self.output_ids = np.full((num_prompts, prompt_len + gen_len), self.config.pad_token_id, dtype=np.int32)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)

        # clear buffers
        L = self.num_layers
        B = self.num_gpu_batches
        for j in range(L):
            for k in range(B):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
                self.partial_cache_read_buf[j][k].clear()
        for j in range(L):
            self.weight_read_buf[j].clear()
            self.partial_weight_read_buf[j].clear()
        for k in range(B):
            self.attention_mask[k].clear()

        self.hidden = array_3d(gen_len, L, B, ValueHolder)

        # init cache after set_task
        for j in range(L):
            for k in range(B):
                self.init_cache(j, k)

        self.generation_loop_normal()

        # delete cache
        for j in range(L):
            for k in range(B):
                self.delete_cache(j, k)

        return self.output_ids


def get_inputs(prompt_len, num_prompts, tokenizer, path):
    prompts = []
    with open(path, "r") as f:
        prompts.append(f.read())
    input_ids = tokenizer(prompts, padding="max_length", max_length=prompt_len).input_ids
    input_ids[0] = input_ids[0][:prompt_len]
    return (input_ids[0],) * num_prompts


def get_filename(args):
    model_size = args.model.split("-")[-1]
    percent = "-".join(str(x) for x in args.percent) + "-"
    filename = (
        f"fo-{model_size}-gbs{args.gpu_batch_size}-ngbs{args.num_gpu_batches}-"
        f"prompt{args.prompt_len}-gen{args.gen_len}-percent-{percent}"
    )
    filename += "cpu-cache" if args.cpu_cache_compute else "gpu-cache"
    if args.compress_weight: filename += "-compw"
    if args.compress_cache: filename += "-compc"
    return filename


def run_flexgen(args):
    print(f"<run_SparsePool>: args.model={args.model}, path={args.path}", flush=True)

    if getattr(args, "path", None) is not None:
        args.path = os.path.expanduser(args.path)
    if getattr(args, "offload_dir", None) is not None:
        args.offload_dir = os.path.expanduser(args.offload_dir)

    model_as_path = os.path.expanduser(args.model) if isinstance(args.model, str) else args.model
    if isinstance(model_as_path, str) and os.path.isdir(model_as_path):
        args.model = model_as_path
        if not (isinstance(args.path, str) and os.path.isdir(args.path)):
            args.path = model_as_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.path, padding_side="left")

    num_prompts = args.num_gpu_batches * args.gpu_batch_size

    warmup_inputs = get_inputs(2048, num_prompts, tokenizer, args.warmup_input_path)
    inputs = get_inputs(args.prompt_len, num_prompts, tokenizer, args.test_input_path)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(
        args.gpu_batch_size, args.num_gpu_batches,
        args.percent[0], args.percent[1],
        args.percent[2], args.percent[3],
        args.percent[4], args.percent[5],
        args.overlap, args.sep_layer, args.pin_weight,
        args.cpu_cache_compute, args.attn_sparsity,
        args.compress_weight,
        CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        args.compress_cache,
        CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
    )

    qwen_config = Qwen3Config.from_pretrained(args.model)
    if qwen_config.pad_token_id is None:
        qwen_config.pad_token_id = 0

    model = OptLM(
        qwen_config, env, args.path, policy,
        partial_weight_ratio=args.partial_weight_ratio,
        alpha=args.alpha,
        max_num_kv=args.max_num_kv,
        gpu_cache_num=args.gpu_cache_num,
        gpu_cache_pred=args.gpu_cache_pred,
        cpu_cache_pred=args.cpu_cache_pred,
    )

    head_dim = getattr(qwen_config, "head_dim",
                       qwen_config.hidden_size // qwen_config.num_attention_heads)
    use_gpu_cache = args.gpu_cache_num != 0
    cache_device = "cuda:0" if use_gpu_cache else "cpu"
    for l in range(qwen_config.num_hidden_layers):
        model._cache_manager.add_cache(
            device=cache_device,
            layer_id=l,
            batch_size=num_prompts,
            head_num=getattr(qwen_config, "num_key_value_heads", qwen_config.num_attention_heads),
            sparse_len=args.max_num_kv,
            hidden_size=head_dim,
            use_gpu_cache=use_gpu_cache,
        )

    use_profile = False  # toggle for torch.profiler runs
    try:
        if use_profile:
            from torch.profiler import profile, ProfilerActivity
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

            print("warmup - generate")
            _ = model.generate(warmup_inputs, max_new_tokens=1, verbose=args.verbose, warmup=True)
            torch.cuda.synchronize()

            print("benchmark - generate")
            timers("generate").reset()
            with profile(activities=activities, with_stack=True) as prof:
                output_ids = model.generate(
                    inputs,
                    max_new_tokens=args.gen_len,
                    debug_mode=args.debug_mode,
                    cut_gen_len=args.cut_gen_len,
                    verbose=args.verbose,
                    warmup=False
                )
            prof.export_chrome_trace(
                f"/root/InfiniGen/speedup/profile_mycache_gpu_b{args.gpu_batch_size}_i{args.prompt_len}_o{args.gen_len}.json"
            )
            costs = timers("generate").costs
        else:
            print("warmup - generate")
            _ = model.generate(warmup_inputs, max_new_tokens=1, verbose=args.verbose, warmup=True)

            print("benchmark - generate")
            timers("generate").reset()
            output_ids = model.generate(
                inputs,
                max_new_tokens=args.gen_len,
                debug_mode=args.debug_mode,
                cut_gen_len=args.cut_gen_len,
                verbose=args.verbose,
                warmup=False
            )
            costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    prefill_latency = costs[0]
    prefill_throughput = num_prompts * args.prompt_len / prefill_latency
    if args.cut_gen_len:
        decode_latency = project_decode_latency(costs, args.prompt_len, args.gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (args.gen_len - 1) / max(decode_latency, 1e-10)
    total_latency = prefill_latency + decode_latency
    total_throughput = (num_prompts * args.gen_len) / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if args.verbose >= 2:
            print("Outputs:\n" + 70 * "-")
            for i in [0, len(outputs) - 1]:
                print(f"{i}: {outputs[i]}")
                print(70 * "-")

    gpu.print_stats()
    cpu.print_stats()

    filename = get_filename(args) + ".log" if args.log_file == "auto" else args.log_file
    log_str = write_benchmark_log(
        filename,
        0, 0, 0,
        gpu_peak_mem, bool(args.debug_mode or args.cut_gen_len),
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput
    )
    if args.verbose >= 1:
        print(log_str)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    print("InfiniGen(Qwen)")
    print(f"input: {args.prompt_len} output: {args.gen_len} bsz: {num_prompts}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Total: {total_latency} Prefill: {prefill_latency} Decode: {decode_latency}")
    print(f"###Log name=InfiniGen input_len={args.prompt_len} output_len={args.gen_len} bsz={num_prompts} total={total_latency} prefill={prefill_latency} decode={decode_latency}")
    print("=================================================")


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model repo id or local HF dir.")
    parser.add_argument("--path", type=str, default="~/llama_weights", help="Path to HF weights dir (safetensors).")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir", help="Offload directory.")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int, help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str, choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=1)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int, default=[100, 0, 100, 0, 100, 0])
    parser.add_argument("--sep-layer", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true")
    parser.add_argument("--compress-cache", action="store_true")
    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--overlap", type=str2bool, nargs="?", const=True, default=True)

    # sparse/prefetch/partial
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--partial-weight-ratio", type=float, default=0.2)
    parser.add_argument("--max-num-kv", type=int, default=400)
    parser.add_argument("--gpu-cache-num", type=int, default=0,
                        help="GPU cache pool 的数量/分片数；为 0 时使用 CPU cache pool")
    parser.add_argument("--gpu-cache-pred", type=float, default=1.0,
                        help="GPU cache 容量倍率，相对于 max-num-kv")
    parser.add_argument("--cpu-cache-pred", type=float, default=1.0,
                        help="CPU cache 容量倍率，相对于 max-num-kv")
 

    parser.add_argument("--warmup-input-path", type=str, required=True)
    parser.add_argument("--test-input-path", type=str, required=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    assert len(args.percent) == 6
    run_flexgen(args)
