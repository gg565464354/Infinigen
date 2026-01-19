"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

import sys

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

from infinigen.skewing_controller import reform_hidden_states, skew, skew_gqa
from infinigen.partial_weight_generation_controller import partial_weight_index_generation
from infinigen.kv_selection_controller import speculate_attention

from flexgen.cache_selection_controller_v2 import reconstruct_unhit_only_on_gpu


general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None

# Some repos use bf16 KV cache. utils.py doesn't map bf16 in a few places; patch it here
# to avoid runtime KeyError in memory accounting.
if torch.bfloat16 not in torch_dtype_to_num_bytes:
    torch_dtype_to_num_bytes[torch.bfloat16] = 2


def _getattr_first(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _get_num_attention_heads(config) -> int:
    v = _getattr_first(config, ["num_attention_heads", "n_head"])
    if v is None:
        raise AttributeError("Config missing num_attention_heads/n_head")
    return int(v)


def _get_num_kv_heads(config) -> int:
    v = _getattr_first(config, ["num_key_value_heads", "num_attention_heads", "n_head"])
    if v is None:
        raise AttributeError("Config missing num_key_value_heads/num_attention_heads/n_head")
    return int(v)


def _get_hidden_size(config) -> int:
    v = _getattr_first(config, ["hidden_size", "input_dim"])
    if v is None:
        raise AttributeError("Config missing hidden_size/input_dim")
    return int(v)


def _get_head_dim(config) -> int:
    v = _getattr_first(config, ["head_dim"])
    if v is not None:
        return int(v)
    hidden_size = _get_hidden_size(config)
    n_head = _get_num_attention_heads(config)
    return int(hidden_size // n_head)


def _get_config_torch_dtype(config) -> torch.dtype:
    dtype = _getattr_first(config, ["dtype", "torch_dtype"])
    if dtype is None:
        return torch.bfloat16
    if isinstance(dtype, str):
        return getattr(torch, dtype, torch.bfloat16)
    return dtype


def _get_cache_torch_dtype(config) -> torch.dtype:
    """
    Cache dtype policy:
    - If model dtype is bf16, use fp16 for KV cache (bf16 causes issues with some custom ops / numpy mapping).
    - Otherwise follow model dtype.
    """
    dtype = _get_config_torch_dtype(config)
    if dtype == torch.bfloat16:
        return torch.bfloat16
    return dtype


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


def speculate_attention_grouped(
    hidden: torch.Tensor,
    p_w_q: torch.Tensor,
    p_k_c: torch.Tensor,
    n_head: int,
    n_kv_head: Optional[int],
    max_num_kv: int,
    group_size: int = 4,
):
    """Speculate indices by aggregating scores across GQA head groups."""
    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))  # (b*n_head, 1, n)
    p_attn = p_attn.squeeze(1).view(b, n_head, -1)

    if n_kv_head is None:
        n_kv_head = n_head
    rep = max(1, n_head // n_kv_head)
    p_attn = p_attn.view(b, n_kv_head, rep, -1).mean(dim=2)  # (b, n_kv, n)
    token_scores = p_attn.reshape(b * n_kv_head, -1)  # (b*n_kv, n)

    n = token_scores.shape[1]
    if n == 0:
        return None, None

    group_size = max(1, int(group_size))
    group_count = (n + group_size - 1) // group_size
    pad_len = group_count * group_size - n
    if pad_len:
        pad = torch.full(
            (token_scores.shape[0], pad_len),
            -float("inf"),
            device=token_scores.device,
            dtype=token_scores.dtype,
        )
        token_scores = torch.cat([token_scores, pad], dim=1)

    group_scores = token_scores.view(token_scores.shape[0], group_count, group_size).max(dim=2).values
    max_groups = max(1, min(max_num_kv // group_size, group_count))
    topk_groups = torch.topk(group_scores, k=max_groups, dim=1).indices  # (b*n_kv, g)

    offsets = torch.arange(group_size, device=token_scores.device)
    token_idx = topk_groups.unsqueeze(-1) * group_size + offsets  # (b*n_kv, g, group_size)
    token_idx = token_idx.reshape(token_scores.shape[0], -1)  # (b*n_kv, L)

    invalid_mask = token_idx >= n
    if invalid_mask.any():
        token_idx = torch.where(invalid_mask, torch.full_like(token_idx, n - 1), token_idx)

    prefetch_idx = token_idx.transpose(0, 1).unsqueeze(1)
    if invalid_mask.any():
        pad_idx = invalid_mask.transpose(0, 1).unsqueeze(1).to(torch.int32)
    else:
        pad_idx = torch.zeros(
            (prefetch_idx.shape[0], 1, prefetch_idx.shape[2]),
            device=prefetch_idx.device,
            dtype=torch.int32,
        )

    return prefetch_idx, pad_idx


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x.unsqueeze(2).expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def repeat_kv_cache(x: torch.Tensor, n_rep: int, n_kv_head: int) -> torch.Tensor:
    """
    x: (s, b * n_kv, d)
    returns: (s, b * n_head, d)
    """
    if n_rep == 1:
        return x
    s, total_kv, d = x.shape
    b = total_kv // n_kv_head
    x = x.view(s, b, n_kv_head, d)
    x = x.unsqueeze(3).expand(s, b, n_kv_head, n_rep, d)
    x = x.reshape(s, b, n_kv_head * n_rep, d)
    return x.reshape(s, b * (n_kv_head * n_rep), d)


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = input.dtype
    variance = input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden_states = input * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


def rms_norm_gqa(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n_head = x.shape[2]
    head_dim = x.shape[3]
    n_kv_head = weight.shape[0] // head_dim
    if n_head % n_kv_head != 0:
        raise ValueError("n_head must be divisible by n_kv_head for GQA RMSNorm")
    rep = n_head // n_kv_head

    x = x.view(x.shape[0], x.shape[1], n_kv_head, rep, head_dim)
    w = weight.view(n_kv_head, head_dim).unsqueeze(0).unsqueeze(2)  # (1, n_kv, 1, d)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps) * w
    return x.view(x.shape[0], x.shape[1], n_head, head_dim)


def rms_norm_with_headwise_weight(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm that supports either per-dim weight (D,) or per-head weight (H*D,)."""
    if weight is None:
        return x
    if weight.numel() == x.shape[-1]:
        return rms_norm(x, weight=weight, eps=eps)

    n_head = x.shape[-2]
    head_dim = x.shape[-1]
    if weight.numel() != n_head * head_dim:
        raise ValueError(
            f"Unsupported RMSNorm weight shape: numel={weight.numel()} expected {head_dim} or {n_head * head_dim}"
        )

    w = weight.view(n_head, head_dim)
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return (x_norm * w).to(x.dtype)


def get_rotary_position_embeddings(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "head_dim must be even"
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_q = q.size(-2)
    if cos.size(-2) != s_q:
        cos = cos[:s_q]
        sin = sin[:s_q]
    cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            if self.dtype not in torch_dtype_to_np_dtype:
                raise KeyError(f"Compressed device allocate doesn't support dtype={self.dtype}")
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        elif dst.device_type == DeviceType.DISK:
            if self.dtype not in torch_dtype_to_np_dtype:
                raise KeyError(f"Disk offload doesn't support dtype={self.dtype}")
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        else:
            # TorchDevice.allocate accepts torch.dtype directly; avoid numpy dtype mapping
            # (numpy has no standard bfloat16, which causes KeyError).
            ret = dst.allocate(shape, self.dtype)
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        torch_dtype = dtype if isinstance(dtype, torch.dtype) else np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=torch_dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        # print("opt_input_embed begin", flush=True)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        
        # print("opt_input_embed embedding begin", flush=True)

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        torch.cuda.synchronize()
        # print("opt_input_embed cumsum begin", flush=True)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]
        positions = positions.to(torch.long) 

        
        torch.cuda.synchronize()
        # print("opt_input_embed pos_embed embedding begin", flush=True)

        ## just for test
        # vocab_size = w_pos.data.size(0)  # 位置 embedding 的长度

        # min_pos = int(positions.min().item())
        # max_pos = int(positions.max().item())

        # print(f"[DEBUG] positions range: {min_pos} ~ {max_pos}, vocab_size={vocab_size}", flush=True)

        # if min_pos < 0 or max_pos >= vocab_size:
        #     raise RuntimeError(
        #         f"positions out of range: min={min_pos}, max={max_pos}, expected [0, {vocab_size-1}]"
        #     )
        ## just for test


        pos_embed = F.embedding(positions, w_pos.data)

        
        torch.cuda.synchronize()
        # print("opt_input_embed pos_embed embedding end", flush=True)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache

    def init_cache_one_gpu_batch_infin(self, config, task, policy):
        num_kv_head, prompt_len, gen_len, gpu_batch_size = (
            _get_num_kv_heads(config),
            task.prompt_len,
            task.gen_len,
            policy.gpu_batch_size,
        )
        head_dim = _get_head_dim(config)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_kv_head, head_dim)
        pin_memory = False
        dtype = _get_cache_torch_dtype(config)
        k_cache = self.allocate(shape, dtype, pin_memory=pin_memory)
        v_cache = self.allocate(shape, dtype, pin_memory=pin_memory)
        return k_cache, v_cache


    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
        w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, warmup=False, partial_weight_ratio=0.1):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        new_h = reform_hidden_states(hidden)

        # shape: (b, s, h)
        q = F.linear(new_h, w_q.data, bias=None)
        k = F.linear(new_h, w_k.data, bias=None)
        v = F.linear(hidden, w_v.data, bias=b_v.data)

        # Partial weight index generation
        # partial_weight_index = None
        # if (not warmup) and (partial_weight_ratio is not None):
        #     partial_weight_index = partial_weight_index_generation(q, n_head, head_dim, partial_weight_ratio)

        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # Generate skewing matrix
        # if warmup:
        #     w_q.data, w_k.data = skew(q, k, w_q.data, w_k.data, n_head, head_dim)

        # ========== ⬇️ 替换为：正确构造 Bool mask + 保证精度 ⬇️ ==========

        # 重塑为标准 SDPA 形状 (b, n_head, s, head_dim)
        q_sdpa = q.permute(0, 2, 1, 3)  # (b, n_head, s, head_dim)
        k_sdpa = k.permute(0, 2, 1, 3)  # (b, n_head, s, head_dim)
        v_sdpa = v.permute(0, 2, 1, 3)  # (b, n_head, s, head_dim)

        # 构造组合 Bool mask
        causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool, device=self.dev)).view(1, 1, s, s)
        key_padding_mask = attention_mask.data.view(b, 1, 1, s)  # (b, 1, 1, s)
        combined_mask = causal_mask & key_padding_mask  # (b, 1, s, s)

        # 使用 Bool mask —— 启用内存高效实现
        value_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=combined_mask,
            dropout_p=0.0,
            is_causal=False
        )

        value = value_sdpa
        # ========== ⬆️ 替换结束 ⬆️ ==========

        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # 保持缓存格式不变
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)  # (b * n_head, head_dim, s)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)  # (b * n_head, s, head_dim)

        k = k.permute(2, 0, 1)  # (s, b * n_head, head_dim)
        v = v.permute(1, 0, 2)  # (s, b * n_head, head_dim)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v, w_q, w_k, None
    
    
    
    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config, quest_selector, topk, speculation_stream, alpha, max_num_kv):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape

        # qin change
        src_s = 0
        if isinstance(k_cache, list):
            cache_len = sum([k.shape[0] for k in k_cache])
            src_s = min(attention_mask.shape[1], cache_len + 1)
        else:
            src_s = min(attention_mask.shape[1], k_cache.shape[0] + 1)
        
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        new_h = reform_hidden_states(hidden)
        
        
        # TODO: generate quest prefetch_id, pad_idx set as None
        # Speculate attention
        prefetch_idx = None
        pad_idx = None
        # if p_w_q is not None:
        #     with torch.cuda.stream(speculation_stream):
        #         prefetch_idx, pad_idx = new_speculate_attention(new_h, p_w_q, partial_k_cache, n_head, alpha, max_num_kv)
        
        

        # shape: (b, 1, h)
        q = F.linear(new_h, w_q.data, bias=None)
        k = F.linear(new_h, w_k.data, bias=None)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)
        
        
        ###################### use the selector of the next deoder layer to get prefetch_idx
        if quest_selector is not None:
            if not hasattr(quest_selector, "quest_select_for_batch"):
                raise TypeError("quest_selector must have method `quest_select_for_batch`")
            if not isinstance(topk, int):
                raise TypeError("topk must be int")
            
            with torch.cuda.stream(speculation_stream):
                # prefetch_idx, pad_idx = new_speculate_attention(new_h, p_w_q, partial_k_cache, n_head, alpha, max_num_kv)
                p_q = q.permute(0, 2, 1, 3)
                prefetch_idx = quest_selector.quest_select_for_batch(p_q, topk)
        
        ###################### use the selector of the next deoder layer to get prefetch_idx
        
        

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s-1]
                    v = v_cache.data[:src_s-1]
                k  = torch.cat((k, k_new), dim = 0)
                v  = torch.cat((v, v_new), dim = 0)

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, -1)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, -1, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, None,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, None,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new, (prefetch_idx, pad_idx)

    # remove p_w_q and partial_k_cache, add as quest_selector and topk
    def patch_mha_gen_with_update(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, group_k_cache, group_v_cache, donate, unhit_id_map,
                attn_sparsity, compress_cache, comp_config, quest_selector, topk, speculation_stream, 
                alpha, max_num_kv, cache_manager, layer_id):
        """Multi-head attention (decoding phase)."""
        # print("patch_mha_gen_with_update begin", flush=True)
        
        # 优先处理KV Cache
        if isinstance(group_k_cache[0], tuple):
            cache_k = group_k_cache[0][0]
            cache_v = group_v_cache[0][0]
            
            unhit_k = group_k_cache[0][1]
            unhit_v = group_v_cache[0][1]
            
            cache_head_dim = cache_k.shape[2]
            
            if unhit_id_map != None and type(unhit_id_map) == torch.Tensor:
                assert unhit_id_map.device == unhit_k.device, f"unhit id map must on {unhit_k.device}"
                
                unhit_k = reconstruct_unhit_only_on_gpu(
                    unhit_id_map_gpu = unhit_id_map,
                    global_unhit_kv_gpu=unhit_k,
                    head_dim=cache_head_dim,
                    device=unhit_k.device
                )
                unhit_v = reconstruct_unhit_only_on_gpu(
                    unhit_id_map_gpu = unhit_id_map,
                    global_unhit_kv_gpu=unhit_v,
                    head_dim=cache_head_dim,
                    device=unhit_k.device
                )
            
            final_group_k = [torch.cat([cache_k, unhit_k], dim=0)]
            final_group_v = [torch.cat([cache_v, unhit_v], dim=0)]

        else:        
            final_group_k = group_k_cache
            final_group_v = group_v_cache
            
        
        # print("patch_mha_gen_with_update #1", flush=True)
        

        # decompress weights
        b, tgt_s, h = inputs.shape

        
        # print("patch_mha_gen_with_update #2", flush=True)
        
        # qin change
        src_s = 0
        if isinstance(final_group_k, list):
            cache_len = sum([k.shape[0] for k in final_group_k])
            src_s = min(attention_mask.shape[1], cache_len + 1)
        else:
            src_s = min(attention_mask.shape[1], final_group_k.shape[0] + 1)
        
        
        # print("patch_mha_gen_with_update #2.1", flush=True)

        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        new_h = reform_hidden_states(hidden)
        
        
        # print("patch_mha_gen_with_update #2.2", flush=True)
        
        # TODO: quest select
        # Speculate attention
        prefetch_idx = None
        pad_idx = None
        
        
        # print("patch_mha_gen_with_update #3", flush=True)

        # shape: (b, 1, h)
        q = F.linear(new_h, w_q.data, bias=None)
        k = F.linear(new_h, w_k.data, bias=None)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)
        
        
        ###################### use the selector of the next deoder layer to get prefetch_idx
        if quest_selector is not None:
            if not hasattr(quest_selector, "quest_select_for_batch"):
                raise TypeError("quest_selector must have method `quest_select_for_batch`")
            if not isinstance(topk, int):
                raise TypeError("topk must be int")
            
            with torch.cuda.stream(speculation_stream):
                # prefetch_idx, pad_idx = new_speculate_attention(new_h, p_w_q, partial_k_cache, n_head, alpha, max_num_kv)
                p_q = q.permute(0, 2, 1, 3)
                prefetch_idx = quest_selector.quest_select_for_batch(p_q, topk)
        
        ###################### use the selector of the next deoder layer to get prefetch_idx
        

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        ########################## Attention 计算部分(可以考虑改进为 flash attention)
        cur_k_cache = final_group_k[0]
        cur_v_cache = final_group_v[0]

        if attn_sparsity >= 1.0:  # Dense attention
            # 对每个 group 单独处理
            
            k  = torch.cat((cur_k_cache, k_new), dim = 0)
            v  = torch.cat((cur_v_cache, v_new), dim = 0)

            # shape: (b * n_head, head_dim, s)
            k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, -1)
            # shape: (b * n_head, s, head_dim)
            v = v.permute(1, 0, 2).reshape(b * n_head, -1, head_dim)

            if k.is_cuda:
                value = self._attention_value(q, k, v, None,
                    b, src_s, tgt_s, n_head, head_dim)
            else:
                raise ValueError("patch_mha_gen Error")


        else:  # Sparse attention, as a compitetion of mask-based attention
            # shape: (s, b * n_head, head_dim)
            raise ValueError("pytorch_backend mha_gen attn_sparsity is error")

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)


        ########################## Attention 计算部分
        
        # 
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)
            
        
        # incremental update the cache manager
        cache_manager.update_gpu_cache_with_new_tensor(layer_id, final_group_k, final_group_v)
        
        return TorchTensor.create_from_torch(value, self), k_new, v_new, (prefetch_idx, pad_idx)


    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        if mask is not None:
            mask = mask.view(b, 1, 1, src_s)
        

        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        if mask is not None:
            attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    ##### qin change
    
    def _group_attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        if mask is not None:
            mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        if mask is not None:
            attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    # only use for decode
    # add layer_group_ids param, use to split q into multiple group
    def _group_attention_value(self, q, group_k, group_v, layer_group_ids, mask, b, src_s, tgt_s, n_head, head_dim):
        final_result = torch.empty((b, n_head, tgt_s, head_dim), dtype=q.dtype)

        for i in range(len(layer_group_ids)):
            id_list = layer_group_ids[i]
            tmp_q = query_tensor.index_select(dim=2, index=id_list)
            tmp_n_head = len(id_list)//b
            
            tmp_k = group_k[i]
            tmp_v = group_v[i]
            sparse_len = group_k[i].shape[0]

            # print("here ?")
            # shape: (b * n_head, 1, s)
            attn_weights = self._attention_weights(tmp_q, tmp_k, mask, b, src_s, tmp_n_head)
            # shape: (b, n_head, 1, head_dim)
            tmp_result = torch.bmm(attn_weights, tmp_v).view(b, tmp_n_head, tgt_s, head_dim)

            final_result[:, id_list, :, :] = tmp_result
        
        return final_result

    
    ##### qin change

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"

    ########################################################
    # Qwen3 kernels (InfiniGen-v2 + optional Quest prefetch)
    ########################################################
    def mha_qwen_infin_v2(
        self,
        inputs,
        attention_mask,
        w_q,
        w_k,
        w_v,
        w_out,
        w_ln,
        q_ln,
        k_ln,
        n_head,
        donate,
        compress_cache,
        comp_config,
        eps,
        head_dim,
        num_key_value_groups,
        num_key_value_heads,
        position_embeddings,
        warmup,
        partial_weight_ratio,
    ):
        x = inputs.data if isinstance(inputs, TorchTensor) else inputs
        mask = attention_mask.data if isinstance(attention_mask, TorchTensor) else attention_mask

        b, s, _ = x.shape
        scaling = head_dim ** -0.5
        cache_dtype = x.dtype

        # RoPE embeddings
        if position_embeddings is not None:
            cos_global, sin_global = position_embeddings
            if cos_global.shape[-1] != head_dim:
                cos, sin = get_rotary_position_embeddings(s, head_dim, cos_global.device, x.dtype)
            else:
                cos, sin = cos_global[:s], sin_global[:s]
        else:
            cos, sin = get_rotary_position_embeddings(s, head_dim, x.device, x.dtype)

        # RMS norm + projections
        hidden = rms_norm(x, weight=w_ln, eps=eps)
        q = F.linear(hidden, w_q)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        # MHA vs GQA
        is_mha = (w_k.shape[0] == n_head * head_dim)
        kv_head = n_head if is_mha else num_key_value_heads

        # Column-wise partial index (used by InfiniGen speculation)
        partial_weight_index = None
        if (not warmup) and (partial_weight_ratio is not None):
            partial_weight_index = partial_weight_index_generation(q, n_head, head_dim, partial_weight_ratio)

        # Reshape to heads
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, kv_head, head_dim)
        v = v.view(b, s, kv_head, head_dim)

        # Q/K norm + scaling
        q = rms_norm_with_headwise_weight(q, q_ln, eps=eps) * scaling
        k = rms_norm_with_headwise_weight(k, k_ln, eps=eps)

        # (b, h, s, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # GQA repeat for attention only
        k_base = k
        v_base = v
        if not is_mha:
            k = repeat_kv(k_base, num_key_value_groups)
            v = repeat_kv(v_base, num_key_value_groups)
        else:
            k = k_base
            v = v_base

        # Warmup skewing (InfiniGen)
        if warmup:
            w_q, w_k = skew_gqa(q, k, w_q, w_k, n_head, num_key_value_heads, head_dim)

        # SDPA: avoid materializing full (s,s) mask when there is no padding.
        is_all_valid = bool(mask.all()) if mask is not None else True
        if is_all_valid:
            value = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            causal_mask = torch.tril(torch.ones((s, s), dtype=torch.bool, device=x.device)).view(1, 1, s, s)
            key_padding_mask = mask.view(b, 1, 1, s)
            combined_mask = causal_mask & key_padding_mask
            value = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=combined_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        value = value.transpose(1, 2).reshape(b, s, n_head * head_dim)
        value = F.linear(value, w_out)
        value.add_(x)

        if donate and len(donate) > 0 and donate[0] and isinstance(inputs, TorchTensor):
            inputs.delete()
        if donate and len(donate) > 1 and donate[1] and isinstance(attention_mask, TorchTensor):
            attention_mask.delete()

        # KV cache stores base (kv_head) keys/values: (s, b*kv, d)
        k_cache = k_base.permute(2, 0, 1, 3).reshape(s, b * kv_head, head_dim)
        v_cache = v_base.permute(2, 0, 1, 3).reshape(s, b * kv_head, head_dim)
        if k_cache.dtype != cache_dtype:
            k_cache = k_cache.to(cache_dtype)
            v_cache = v_cache.to(cache_dtype)

        if compress_cache:
            k_cache = self.compressed_device.compress(k_cache, comp_config)
            v_cache = self.compressed_device.compress(v_cache, comp_config)
        else:
            k_cache = TorchTensor.create_from_torch(k_cache, self)
            v_cache = TorchTensor.create_from_torch(v_cache, self)

        return TorchTensor.create_from_torch(value, self), k_cache, v_cache, w_q, w_k, partial_weight_index

    def mha_gen_qwen_infin_v2(
        self,
        inputs,
        attention_mask,
        w_q,
        w_k,
        w_v,
        w_out,
        w_ln,
        q_ln,
        k_ln,
        n_head,
        k_cache,
        v_cache,
        donate,
        attn_sparsity,
        compress_cache,
        comp_config,
        eps,
        head_dim,
        num_key_value_groups,
        num_key_value_heads,
        position_embeddings,
        p_w_q,
        partial_k_cache,
        speculation_stream,
        alpha,
        max_num_kv,
        attn_topk=None,
        prefetch_grouped=False,
        quest_selector=None,
        quest_topk: Optional[int] = None,
    ):
        x = inputs.data if isinstance(inputs, TorchTensor) else inputs
        mask = attention_mask.data if isinstance(attention_mask, TorchTensor) else attention_mask

        b, tgt_s, _ = x.shape
        src_s = mask.shape[1]
        scaling = head_dim ** -0.5
        cache_dtype = None

        # Cache length (may be sparse)
        if isinstance(k_cache, TorchTensor):
            cache_len = k_cache.data.shape[0]
            cache_dtype = k_cache.data.dtype
        else:
            cache_len = k_cache.shape[0]
            cache_dtype = k_cache.dtype
        if cache_dtype is None:
            cache_dtype = x.dtype

        # RMS norm
        hidden = rms_norm(x, weight=w_ln, eps=eps)

        # Prefetch idx generation
        prefetch_idx = None
        pad_idx = None

        if (quest_selector is None) and (p_w_q is not None):
            if speculation_stream is None:
                if prefetch_grouped:
                    prefetch_idx, pad_idx = speculate_attention_grouped(
                        hidden, p_w_q, partial_k_cache, n_head, num_key_value_heads, max_num_kv
                    )
                else:
                    prefetch_idx = speculate_attention(hidden, p_w_q, partial_k_cache, n_head, alpha, max_num_kv)
            else:
                with torch.cuda.stream(speculation_stream):
                    if prefetch_grouped:
                        prefetch_idx, pad_idx = speculate_attention_grouped(
                            hidden, p_w_q, partial_k_cache, n_head, num_key_value_heads, max_num_kv
                        )
                    else:
                        prefetch_idx = speculate_attention(hidden, p_w_q, partial_k_cache, n_head, alpha, max_num_kv)

        # QKV projections
        q = F.linear(hidden, w_q)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        is_mha = (w_k.shape[0] == n_head * head_dim)
        kv_head = n_head if is_mha else num_key_value_heads
        rep = n_head // kv_head

        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, kv_head, head_dim)
        v = v.view(b, tgt_s, kv_head, head_dim)

        q = rms_norm_with_headwise_weight(q, q_ln, eps=eps) * scaling
        k = rms_norm_with_headwise_weight(k, k_ln, eps=eps)

        q = q.transpose(1, 2)  # (b, n_head, 1, d)
        k = k.transpose(1, 2)  # (b, n_kv, 1, d)
        v = v.transpose(1, 2)  # (b, n_kv, 1, d)
        # Make q/k/v match cache dtype for attention (avoid casting full cache).
        if q.dtype != cache_dtype:
            q = q.to(cache_dtype)
        if k.dtype != cache_dtype:
            k = k.to(cache_dtype)
            v = v.to(cache_dtype)

        # RoPE for the last position only.
        # IMPORTANT: keep cos/sin in the same dtype as q/k; otherwise (fp16 * bf16) promotes to fp32,
        # and SDPA will error due to dtype mismatch with v (fp16).
        position_id = src_s - 1
        rope_compute_dtype = torch.float32
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=rope_compute_dtype) / head_dim)
        )
        t = torch.tensor([position_id], device=x.device, dtype=rope_compute_dtype)
        freqs = torch.outer(t, inv_freq).repeat_interleave(2, dim=-1)
        cos = freqs.cos().to(q.dtype)
        sin = freqs.sin().to(q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # Quest prefetch (optional)
        if quest_selector is not None:
            if quest_topk is None:
                raise ValueError("quest_topk is required when quest_selector is provided")
            stream_ctx = torch.cuda.stream(speculation_stream) if speculation_stream is not None else None
            if stream_ctx is None:
                q_for_select = q
                if getattr(quest_selector, "num_heads", None) is not None and quest_selector.num_heads != q_for_select.shape[1]:
                    if num_key_value_heads is None:
                        q_for_select = q_for_select[:, : quest_selector.num_heads]
                    else:
                        rep_q = q_for_select.shape[1] // num_key_value_heads
                        q_for_select = q_for_select.reshape(b, num_key_value_heads, rep_q, tgt_s, head_dim).mean(dim=2)
                prefetch_idx = quest_selector.quest_select_for_batch(q_for_select, quest_topk)
                pad_idx = None
            else:
                with stream_ctx:
                    q_for_select = q
                    if getattr(quest_selector, "num_heads", None) is not None and quest_selector.num_heads != q_for_select.shape[1]:
                        if num_key_value_heads is None:
                            q_for_select = q_for_select[:, : quest_selector.num_heads]
                        else:
                            rep_q = q_for_select.shape[1] // num_key_value_heads
                            q_for_select = q_for_select.reshape(b, num_key_value_heads, rep_q, tgt_s, head_dim).mean(dim=2)
                    prefetch_idx = quest_selector.quest_select_for_batch(q_for_select, quest_topk)
                    pad_idx = None

        # New KV for cache update: (1, b * n_kv, d)
        k_cache_new = k.permute(2, 0, 1, 3).reshape(tgt_s, b * kv_head, head_dim)
        v_cache_new = v.permute(2, 0, 1, 3).reshape(tgt_s, b * kv_head, head_dim)

        use_sparse = (attn_topk is not None) or (attn_sparsity < 1.0)

        if not isinstance(k_cache, TorchTensor):
            raise NotImplementedError("Mixed device attention not supported for Qwen3 path")

        if not use_sparse:
            if compress_cache:
                k_all = k_cache.device.decompress(k_cache)[:cache_len]
                v_all = v_cache.device.decompress(v_cache)[:cache_len]
            else:
                k_all = k_cache.data[:cache_len]
                v_all = v_cache.data[:cache_len]

            k_all = torch.cat([k_all, k_cache_new], dim=0)
            v_all = torch.cat([v_all, v_cache_new], dim=0)

            if rep > 1:
                k_all = repeat_kv_cache(k_all, rep, kv_head)
                v_all = repeat_kv_cache(v_all, rep, kv_head)

            k_all = k_all.permute(1, 0, 2).view(b, n_head, cache_len + 1, head_dim)
            v_all = v_all.permute(1, 0, 2).view(b, n_head, cache_len + 1, head_dim)

            value = F.scaled_dot_product_attention(
                q, k_all, v_all,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            )
            value = value.view(b, n_head, tgt_s, head_dim).transpose(1, 2).reshape(b, tgt_s, n_head * head_dim)
        else:
            if compress_cache:
                k_base = k_cache.device.decompress(k_cache)
                v_base = v_cache.device.decompress(v_cache)
            else:
                k_base = k_cache.data
                v_base = v_cache.data

            if k_base.shape[0] >= src_s:
                k_hist = k_base[:src_s]
                v_hist = v_base[:src_s]
                k_hist[src_s - 1 : src_s] = k_cache_new
                v_hist[src_s - 1 : src_s] = v_cache_new
                local_s = src_s
                local_mask = mask[:, :local_s].view(b, 1, 1, local_s)
            else:
                k_hist = torch.cat([k_base, k_cache_new], dim=0)
                v_hist = torch.cat([v_base, v_cache_new], dim=0)
                local_s = k_hist.shape[0]
                local_mask = torch.ones((b, 1, 1, local_s), dtype=torch.bool, device=x.device)

            if rep > 1:
                k_hist = repeat_kv_cache(k_hist, rep, kv_head)
                v_hist = repeat_kv_cache(v_hist, rep, kv_head)

            k_all = k_hist.view(local_s, b, n_head, head_dim).permute(1, 2, 0, 3)
            v_all = v_hist.view(local_s, b, n_head, head_dim).permute(1, 2, 0, 3)

            scores = torch.matmul(q.to(torch.float32), k_all.transpose(-1, -2).to(torch.float32)) / (head_dim ** 0.5)
            scores = scores.masked_fill(~local_mask, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)

            hist_probs = probs[..., :-1]
            max_history = hist_probs.shape[-1]
            if max_history <= 0:
                weight_last = probs[..., -1:]
                attn_out = v_all[:, :, -1:, :].to(torch.float32) * weight_last.to(torch.float32)
            else:
                if attn_topk is not None:
                    topk = max(1, min(int(attn_topk), max_history))
                else:
                    topk = int(attn_sparsity * max_history)
                    topk = max(1, min(topk, max_history))

                topk_weights, topk_indices = torch.topk(hist_probs, k=topk, dim=-1, sorted=False)
                v_hist = v_all[:, :, :-1, :]
                gather_idx = topk_indices.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, head_dim)
                v_sel = torch.gather(v_hist, dim=2, index=gather_idx)
                v_sel = torch.cat([v_sel, v_all[:, :, -1:, :]], dim=2)

                weights = torch.cat([topk_weights.squeeze(2), probs[..., -1:].squeeze(2)], dim=-1)
                attn_out = torch.sum(v_sel.to(torch.float32) * weights.to(torch.float32).unsqueeze(-1), dim=2).unsqueeze(2)

            value = attn_out.to(x.dtype).view(b, n_head, tgt_s, head_dim).transpose(1, 2).reshape(b, tgt_s, n_head * head_dim)

        # Project in model dtype for stability/compat, then residual.
        value = value.to(x.dtype)
        value = F.linear(value, w_out)
        value.add_(x)

        if donate and len(donate) > 0 and donate[0] and isinstance(inputs, TorchTensor):
            inputs.delete()
        if donate and len(donate) > 1 and donate[1] and isinstance(attention_mask, TorchTensor):
            attention_mask.delete()

        if compress_cache:
            k_cache_new = self.compressed_device.compress(k_cache_new, comp_config)
            v_cache_new = self.compressed_device.compress(v_cache_new, comp_config)
        else:
            if k_cache_new.dtype != cache_dtype:
                k_cache_new = k_cache_new.to(cache_dtype)
                v_cache_new = v_cache_new.to(cache_dtype)
            k_cache_new = TorchTensor.create_from_torch(k_cache_new, self)
            v_cache_new = TorchTensor.create_from_torch(v_cache_new, self)

        if prefetch_grouped:
            if prefetch_idx is None:
                return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new, None
            return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new, (prefetch_idx, pad_idx)

        return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new, prefetch_idx


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        # tmp qin change
        for t in self.copy_threads:
            t.start()
        # tmp qin change

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        # Disk-backed tensors are stored via NumPy memmap, so `dtype` must be a
        # NumPy dtype. Convert from torch dtype if possible.
        if isinstance(dtype, torch.dtype):
            if dtype not in torch_dtype_to_np_dtype:
                raise ValueError(
                    f"TorchDisk does not support dtype {dtype}. "
                    "Use float16/float32 or disable disk offloading."
                )
            np_dtype = torch_dtype_to_np_dtype[dtype]
            torch_dtype = dtype
        else:
            np_dtype = dtype
            torch_dtype = np_dtype_to_torch_dtype[dtype]

        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=np_dtype)
        return TorchTensor(shape, torch_dtype, path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def init_cache_one_gpu_batch_infin(self, config, task, policy):
        num_kv_head, prompt_len, gen_len, gpu_batch_size = (
            _get_num_kv_heads(config),
            task.prompt_len,
            task.gen_len,
            policy.gpu_batch_size,
        )
        head_dim = _get_head_dim(config)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_kv_head, head_dim)
        dtype = _get_cache_torch_dtype(config)
        k_cache = self.allocate(shape, dtype)
        v_cache = self.allocate(shape, dtype)
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
            
        # tmp qin change
        for t in self.copy_threads:
            t.join()
        # tmp qin change

        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        torch_dtype = dtype if isinstance(dtype, torch.dtype) else np_dtype_to_torch_dtype[dtype]
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, torch_dtype, (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache

    def init_cache_one_gpu_batch_infin(self, config, task, policy):
        num_kv_head, prompt_len, gen_len, gpu_batch_size = (
            _get_num_kv_heads(config),
            task.prompt_len,
            task.gen_len,
            policy.gpu_batch_size,
        )
        head_dim = _get_head_dim(config)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_kv_head, head_dim)

        # Round to multiples of num_kv_head
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_kv_head * num_kv_head
            len_cpu = shape[SEG_DIM] - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_kv_head * num_kv_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_kv_head * num_kv_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        dtype = _get_cache_torch_dtype(config)
        k_cache = self.allocate(shape, dtype, seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, dtype, seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.bfloat16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
