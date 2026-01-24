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

### for infinigen
from infinigen.skewing_controller import reform_hidden_states, skew, skew_gqa
from infinigen.partial_weight_generation_controller import partial_weight_index_generation
from infinigen.kv_selection_controller import speculate_attention

# from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

if torch.bfloat16 not in torch_dtype_to_num_bytes:
    torch_dtype_to_num_bytes[torch.bfloat16] = 2

# from quest.quest_sparse import QuestSparseKVManagerFlat


general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None

# Optional override for link I/O time (debugging / simulation)
force_io_time = None


def speculate_attention_grouped(hidden, p_w_q, p_k_c, n_head, n_kv_head, max_num_kv, group_size=4):
    """Speculates token indices by averaging scores across head groups and selecting top groups."""
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
        pad = torch.full((token_scores.shape[0], pad_len), -float("inf"),
                         device=token_scores.device, dtype=token_scores.dtype)
        token_scores = torch.cat([token_scores, pad], dim=1)

    group_scores = token_scores.view(token_scores.shape[0], group_count, group_size).max(dim=2).values
    max_groups = max(1, min(max_num_kv // group_size, group_count))
    topk_groups = torch.topk(group_scores, k=max_groups, dim=1).indices  # (b*n_kv, g)

    offsets = torch.arange(group_size, device=token_scores.device)
    token_idx = topk_groups.unsqueeze(-1) * group_size + offsets  # (b, g, G)
    token_idx = token_idx.reshape(token_scores.shape[0], -1)  # (b*n_kv, L)

    invalid_mask = token_idx >= n
    if invalid_mask.any():
        token_idx = torch.where(invalid_mask, torch.full_like(token_idx, n - 1), token_idx)

    prefetch_idx = token_idx.transpose(0, 1).unsqueeze(1)

    if invalid_mask.any():
        pad_idx = invalid_mask.transpose(0, 1).unsqueeze(1).to(torch.int32)
    else:
        pad_idx = torch.zeros((prefetch_idx.shape[0], 1, prefetch_idx.shape[2]),
                              device=prefetch_idx.device, dtype=torch.int32)

    return prefetch_idx, pad_idx

# def repeat_kv(x: torch.Tensor, n_rep: int):
#     """Repeats key/value tensors along the head dimension.
    
#     Args:
#         x: (b, s, n_kv_head, d)
#         n_rep: int
    
#     Returns:
#         (b, s, n_kv_head * n_rep, d)
#     """
#     if n_rep == 1:
#         return x
#     b, s, n_kv, d = x.shape
#     return x.unsqueeze(-2).expand(b, s, n_kv, n_rep, d).reshape(b, s, n_kv * n_rep, d)

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x.unsqueeze(2)
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )

# def repeat_kv_cache(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     slen, num_key_value_heads, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :].expand(slen, num_key_value_heads, n_rep, head_dim)
#     return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)

def repeat_kv_cache(x: torch.Tensor, n_rep: int, n_kv_head: int):
    """
    x: (s, b * n_kv, d)
    n_rep: int
    n_kv_head: int
    returns: (s, b * n_head, d)
    """
    if n_rep == 1:
        return x
    s, total_kv, d = x.shape
    b = total_kv // n_kv_head
    x = x.view(s, b, n_kv_head, d)
    x = x.unsqueeze(3).expand(s, b, n_kv_head, n_rep, d)
    x = x.reshape(s, b, n_kv_head * n_rep, d)
    x = x.reshape(s, b * (n_kv_head * n_rep), d)
    return x

def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    input_dtype = input.dtype
    # 计算 RMS（Root Mean Square）
    variance = input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    # 归一化
    hidden_states = input * torch.rsqrt(variance + eps)
    # 缩放
    return (weight * hidden_states).to(input_dtype)

def rms_norm_gqa(x, weight, eps=1e-6):
    """
    RMSNorm for GQA-expanded tensor.
    
    Args:
        x: (b, s, n_head, d)
        weight: (n_kv_head * d,)  ← original GQA norm weight
        eps: epsilon
    
    Returns:
        normalized x: (b, s, n_head, d)
    """
    n_head = x.shape[2]
    n_kv_head = weight.shape[0] // x.shape[3]  # weight.size = n_kv_head * d
    assert n_head % n_kv_head == 0
    rep = n_head // n_kv_head
    head_dim = x.shape[3]

    # Reshape x: (b, s, n_head, d) -> (b, s, n_kv_head, rep, d)
    x = x.view(x.shape[0], x.shape[1], n_kv_head, rep, head_dim)

    # Reshape weight: (n_kv_head * d,) -> (n_kv_head, 1, 1, d)
    #                                → will broadcast to (n_kv_head, rep, d)
    w = weight.view(n_kv_head, head_dim)  # (n_kv_head, d)
    w = w.unsqueeze(1).unsqueeze(0)  # (1, n_kv_head, 1, d)

    # Norm
    variance = x.pow(2).mean(dim=-1, keepdim=True)  # (b, s, n_kv_head, rep, 1)
    x = x * torch.rsqrt(variance + eps) * w  # ✅ broadcast here

    # Reshape back
    x = x.view(x.shape[0], x.shape[1], n_head, head_dim)
    return x


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice
    
# custom rope
def get_rotary_position_embeddings(seq_len: int, head_dim: int, device: torch.device, dtype=torch.bfloat16):
    """
    生成适用于 head_dim 的 cos/sin，保持 dtype 一致
    """
    assert head_dim % 2 == 0, "head_dim must be even"

    # 使用指定 dtype（如 bfloat16）生成
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)  # [0, 1, ..., seq_len-1]

    # 外积：(seq_len, head_dim//2)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim//2)

    # 扩展为 (seq_len, head_dim)
    cos = freqs.cos().repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1)  # (seq_len, head_dim)

    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q: (b, nh, s, d)
    k: (b, nh, s, d)
    cos: (s, d)
    sin: (s, d)
    """
    s_q = q.size(2)  # 当前序列长度
    s_cos = cos.size(-2)  # cos 的序列长度
    
    if s_q != s_cos:
        # 切片 cos/sin 到当前长度
        cos = cos[:s_q]  # (s_q, head_dim)
        sin = sin[:s_q]  # (s_q, head_dim)
    
    cos = cos[None, None, :, :]  # (1, 1, s, d)
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


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
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            # ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
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
        # `dtype` can be either a numpy dtype (used throughout FlexGen) or a
        # torch dtype (used by some InfiniGen/Qwen paths). Do not index the
        # numpy->torch map with a torch dtype.
        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            torch_dtype = np_dtype_to_torch_dtype[dtype]
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
                k_cache = self.allocate(shape, torch.bfloat16, pin_memory=False)
                v_cache = self.allocate(shape, torch.bfloat16, pin_memory=False)
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

    def opt_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            # w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data)
        # print(token_embed[:,:,:6])
        # remove pos embedding
        # positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # # cut positions if `past_key_values_length` is > 0
        # past_key_values_length = mask.shape[1] - token_ids.shape[1]
        # positions = positions[:, past_key_values_length:]

        # pos_embed = F.embedding(positions, w_pos.data)

        # data = token_embed + pos_embed
        data = token_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, w_token, donate,
                         do_sample, temperature, eps):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
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
            config.num_key_value_heads, config.hidden_size, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, config.head_dim)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, torch.bfloat16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, torch.bfloat16, pin_memory=pin_memory)
        return k_cache, v_cache
    
    def init_cache_one_gpu_batch_infin(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            config.hidden_size, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, config.head_dim)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, torch.bfloat16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, torch.bfloat16, pin_memory=pin_memory)
        return k_cache, v_cache

    def mha(self, inputs, attention_mask, w_q, w_k, w_v,
            w_out, w_ln, q_ln, k_ln, n_head, donate, compress_cache, comp_config, eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        # print(inputs.data[:,:,:6])
        b, s, h = inputs.shape
        # head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # print(hidden[:, :, :6])

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data)
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)

        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)

        k = k.view(b, s, num_key_value_heads, head_dim)
        v_single = v.view(b, s, num_key_value_heads, head_dim)

        # layer_norm of q,k
        q = rms_norm(q, weight=q_ln.data, eps=eps) * scaling
        k_single = rms_norm(k, weight=k_ln.data, eps=eps)

        cos, sin = position_embeddings
        q, k_single = apply_rotary_pos_emb(q, k_single, cos, sin, unsqueeze_dim=2)



        k = repeat_kv(k_single, num_key_value_groups)
        v = repeat_kv(v_single, num_key_value_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # origin (b, s, h, d), target shape (s, b*num_key_value_heads, head_dim)
        k = k_single.permute(1,0,2,3).reshape(s, b*num_key_value_heads, head_dim)
        v = v_single.permute(1,0,2,3).reshape(s, b*num_key_value_heads, head_dim)
        # (s, b * n_head, head_dim)
        # k = k.permute(2, 0, 1)
        # v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    def mha_gen(self, inputs, attention_mask, w_q, w_k, w_v,
                w_out,  w_ln,  q_ln, k_ln, n_head,  k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config, eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings, attn_topk=None):
        """Multi-head attention (decoding phase)."""
        # decompress weights

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        # head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, num_key_value_heads, head_dim)
        v_single = v.view(b, tgt_s, num_key_value_heads, head_dim)

        # layer_norm of q,k
        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k_single = rms_norm(k, weight=k_ln, eps=eps)

        cos, sin = position_embeddings
        q, k_single = apply_rotary_pos_emb(q, k_single, cos, sin, unsqueeze_dim=2)

        k = repeat_kv(k_single, num_key_value_groups)
        v = repeat_kv(v_single, num_key_value_groups)



        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        use_sparse = (attn_topk is not None) or (attn_sparsity < 1.0)

        if isinstance(k_cache, TorchTensor):
            if not use_sparse:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]

                k = repeat_kv_cache(k, num_key_value_groups)
                v = repeat_kv_cache(v, num_key_value_groups)

                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
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
                        attn_sparsity, attn_topk)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity, attn_topk).cuda().half()
        else:  # Mixed device attention
            if use_sparse:
                raise NotImplementedError("Sparse attention not supported on mixed device cache")
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # origin (b, s, h, d), target shape (s, b*n_head, head_dim)
        k_new = k_single.permute(1,0,2,3).reshape(tgt_s, b*num_key_value_heads, head_dim)
        v_new = v_single.permute(1,0,2,3).reshape(tgt_s, b*num_key_value_heads, head_dim)

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

        return TorchTensor.create_from_torch(value, self), k_new, v_new
    
    ######################################################## kernal definition for Qwen3
    def mha_qwen(self, inputs, attention_mask, w_q, w_k, w_v,
             w_out, w_ln, q_ln, k_ln, n_head, donate, compress_cache, comp_config,
             eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings):
        """Multi-head attention for Qwen3 (prefill phase)."""
        b, s, h = inputs.shape
        scaling = head_dim ** -0.5

        # Get position embeddings
        if position_embeddings is not None:
            cos_global, sin_global = position_embeddings
            if cos_global.shape[-1] != head_dim:
                cos, sin = get_rotary_position_embeddings(s, head_dim, cos_global.device, inputs.dtype)
            else:
                cos, sin = cos_global[:s], sin_global[:s]  # slice to current length
        else:
            cos, sin = get_rotary_position_embeddings(s, head_dim, inputs.device, inputs.dtype)
            
        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)

        # Linear projections
        # hidden = inputs.data
        q = F.linear(hidden, w_q)  # (b, s, h)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        # Reshape: (b, s, h) -> (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, num_key_value_heads, head_dim)
        v = v.view(b, s, num_key_value_heads, head_dim)

        # Apply Q/K Norm
        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm(k, weight=k_ln, eps=eps)

        # ✅ 转置为 (b, n_head, s, head_dim)
        q = q.transpose(1, 2)  # → (b, n_head, s, head_dim)
        k = k.transpose(1, 2)  # → (b, n_kv, s, head_dim)
        v = v.transpose(1, 2)  # → (b, n_kv, s, head_dim)
        
        
        # ✅ 使用 HF 的 apply_rotary_pos_emb (unsqueeze_dim=1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)  # (b, n_head, s, d)
        
        
        

        # print(f"mha before q shape = {q.shape} dtype = {q.dtype}", flush=True)
        # print(f"mha before k shape = {k.shape} dtype = {k.dtype}", flush=True)
        
        # ✅ GQA: repeat k/v after RoPE
        k_before_repeat = k
        v_before_repeat = v
        k = repeat_kv(k, num_key_value_groups)  # (b, n_head, s, d) -> (b, n_head*rep, s, d)
        v = repeat_kv(v, num_key_value_groups)

        # print(f"mha after q shape = {q.shape} dtype = {q.dtype}", flush=True)
        # print(f"mha after k shape = {k.shape} dtype = {k.dtype}", flush=True)
        
        # Memory-efficient attention (avoids materializing (b, n_head, s, s))
        # NOTE: assumes no padding in attention_mask (common for fixed-length benchmarking).
        if attention_mask is not None and not bool(attention_mask.data.all().item()):
            raise NotImplementedError(
                "Prefill attention with padding mask is not supported in this backend; "
                "please provide non-padded inputs or implement a proper attn_mask."
            )
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        value = value.transpose(1, 2).reshape(b, s, n_head*head_dim)  # (b, s, n_head*head_dim)
        value = F.linear(value, w_out)
        value.add_(inputs.data)  # Residual

        # Donate
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # ✅ Prepare KV Cache: 保存原始 k/v (before repeat)
        # k: (b, n_kv, s, d) -> (s, b * n_kv, d)
        k_cache = k_before_repeat.transpose(0, 2).transpose(1, 3).reshape(s, b * num_key_value_heads, head_dim)
        v_cache = v_before_repeat.transpose(0, 2).transpose(1, 3).reshape(s, b * num_key_value_heads, head_dim)

        if compress_cache:
            k_cache = self.compressed_device.compress(k_cache, comp_config)
            v_cache = self.compressed_device.compress(v_cache, comp_config)
        else:
            k_cache = TorchTensor.create_from_torch(k_cache, self)
            v_cache = TorchTensor.create_from_torch(v_cache, self)

        return TorchTensor.create_from_torch(value, self), k_cache, v_cache
    
    def mha_gen_qwen(self, inputs, attention_mask, w_q, w_k, w_v,
                 w_out, w_ln, q_ln, k_ln, n_head,
                 k_cache, v_cache, donate,
                 attn_sparsity, compress_cache, comp_config,
                 eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings):
        """Multi-head attention for Qwen3 (decoding phase)."""
        b, tgt_s, h = inputs.shape
        d = head_dim
        src_s = attention_mask.shape[1]
        scaling = head_dim ** -0.5
        
        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)
        
        # Project
        # hidden = inputs.data
        q = F.linear(hidden, w_q)  # (b, 1, h)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        # Reshape
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, num_key_value_heads, head_dim)
        v = v.view(b, tgt_s, num_key_value_heads, head_dim)

        # Q/K Norm
        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm(k, weight=k_ln, eps=eps)

        # Transpose
        q = q.transpose(1, 2)  # (b, n_head, 1, head_dim)
        k = k.transpose(1, 2)  # (b, n_kv, 1, head_dim)
        v = v.transpose(1, 2)  # (b, n_kv, 1, head_dim)
        
        
        # print(f"first k = {k.shape}", flush=True)
        # print(f"first v = {v.shape}", flush=True)

        # RoPE
        # get position embedding
        position_id = src_s - 1  # int
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=inputs.device, dtype=inputs.dtype) / head_dim))
        t = torch.tensor([position_id], device=inputs.device, dtype=inputs.dtype)
        freqs = torch.outer(t, freqs)  # (1, head_dim//2)
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # (1, head_dim)
        sin = freqs.sin().repeat_interleave(2, dim=-1)  # (1, head_dim)
        
        # cos, sin = position_embeddings
        # cos = cos[src_s - 1:src_s]  # 只取最后一个位置
        # sin = sin[src_s - 1:src_s]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # GQA
        # k_before_repeat = k
        # v_before_repeat = v
        # k = repeat_kv(k, num_key_value_groups)
        # v = repeat_kv(v, num_key_value_groups)
        k_new = k # (b, n_head, s, d)
        v_new = v
        
        # print(f"before k_new = {k_new.shape}", flush=True)
        # print(f"before v_new = {v_new.shape}", flush=True)

        # Current k/v
        k_new = k.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)
        v_new = v.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)
        
        
        # print(f"after k_new = {k_new.shape}", flush=True)
        # print(f"after v_new = {v_new.shape}", flush=True)

        # Handle cache
        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:
                if compress_cache:
                    k_all = k_cache.device.decompress(k_cache)[:src_s]  # (s, b * n_kv, d)
                    v_all = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    k_all = k_cache.data[:src_s]  # (s, b * n_kv, d)
                    v_all = v_cache.data[:src_s]

                # 更新最后一个 token
                k_all[src_s - 1:src_s] = k_new  # k_new: (1, b * n_kv, d)
                v_all[src_s - 1:src_s] = v_new

                # ✅ GQA: repeat after update
                k_all = repeat_kv_cache(k_all, num_key_value_groups, num_key_value_heads)  # (s, b * n_head, d)
                v_all = repeat_kv_cache(v_all, num_key_value_groups, num_key_value_heads)

                # ✅ 转置为 bmm 友好格式
                k_all = k_all.permute(1, 2, 0)  # (b * n_head, d, s)
                v_all = v_all.permute(1, 0, 2)  # (b * n_head, s, d)

                # Attention
                q_t = q.transpose(1, 2)  # (b, 1, n_head, d) -> (b, n_head, 1, d)
                q_t = q_t.reshape(b * n_head, 1, head_dim)  # (b * n_head, 1, d)

                attn_scores = torch.bmm(q_t, k_all) / (head_dim ** 0.5)  # (b * n_head, 1, s)
                attn_probs = F.softmax(attn_scores, dim=-1)  # (b * n_head, 1, s)
                attn_output = torch.bmm(attn_probs, v_all)  # (b * n_head, 1, d)

                attn_output = attn_output.view(b, n_head, 1, d).transpose(1, 2).reshape(b, 1, n_head*head_dim)
            else:
                raise NotImplementedError("Sparse attention not supported")

        # Reshape output
        # attn_output = attn_output.transpose(1, 2).view(b, tgt_s, h)
        attn_output = F.linear(attn_output, w_out)
        attn_output.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # Create new cache entries (original kv, not repeated)
        # k_new = k_new.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)
        # v_new = v_new.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)

        if compress_cache:
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(attn_output, self), k_new, v_new
    
    ######################################################## 
    
    
    
    ######################################################## qwen mha for infinigen
    def mha_qwen_infin(self, inputs, attention_mask, w_q, w_k, w_v,
                   w_out, w_ln, q_ln, k_ln, n_head, donate, compress_cache, comp_config,
                   eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings,
                   warmup, partial_weight_ratio):
        """Multi-head attention for Qwen3 with InfiniGen's warmup, skew, and column-wise partial cache support."""
        b, s, h = inputs.shape
        scaling = head_dim ** -0.5

        # Decompress weights if compressed

        # Get position embeddings
        if position_embeddings is not None:
            cos_global, sin_global = position_embeddings
            if cos_global.shape[-1] != head_dim:
                cos, sin = get_rotary_position_embeddings(s, head_dim, cos_global.device, inputs.dtype)
            else:
                cos, sin = cos_global[:s], sin_global[:s]
        else:
            cos, sin = get_rotary_position_embeddings(s, head_dim, inputs.device, inputs.dtype)

        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)

        # Linear projections
        q = F.linear(hidden, w_q)  # (b, s, h)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)
        
        # judge the k cache size
        is_mha = False
        if w_k.shape[0] == n_head*head_dim:
            is_mha = True
        
        # Partial weight index: column-wise selection (d' = int(head_dim * ratio))
        partial_weight_index = None
        if (not warmup) and (partial_weight_ratio is not None):
            partial_weight_index = partial_weight_index_generation(q, n_head, head_dim, partial_weight_ratio)

        # Reshape: (b, s, h) -> (b, s, n_head/n_kv, head_dim)
        q = q.view(b, s, n_head, head_dim)
        kv_head = n_head if is_mha else num_key_value_heads
        k = k.view(b, s, kv_head, head_dim)
        v = v.view(b, s, kv_head, head_dim)

        # Q/K Norm
        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm(k, weight=k_ln, eps=eps)

        # Transpose: (b, s, n_head/n_kv, d) -> (b, n_head/n_kv, s, d)
        q = q.transpose(1, 2)  # (b, n_head, s, d)
        k = k.transpose(1, 2)  # (b, n_kv, s, d)
        v = v.transpose(1, 2)  # (b, n_kv, s, d)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # GQA: repeat k/v for attention only, keep base for cache
        k_base = k
        v_base = v
        if not is_mha:
            k = repeat_kv(k_base, num_key_value_groups)
            v = repeat_kv(v_base, num_key_value_groups)
        else:
            k = k_base
            v = v_base

        # Skew mechanism during warmup
        if warmup:
            w_q.data, w_k.data = skew_gqa(q, k, w_q, w_k, n_head, num_key_value_heads, head_dim)
        
        # Memory-efficient attention (avoids materializing (b, n_head, s, s))
        if attention_mask is not None and not bool(attention_mask.data.all().item()):
            raise NotImplementedError(
                "Prefill attention with padding mask is not supported in this backend; "
                "please provide non-padded inputs or implement a proper attn_mask."
            )
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        # value = value.transpose(1, 2).reshape(b, s, h)  # (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, n_head*head_dim)  # (b, s, n_head*d)
        value = F.linear(value, w_out)
        value.add_(inputs.data)  # Residual

        # Donate
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # KV Cache: (s, b * n_kv_head, head_dim)
        k_cache = k_base.permute(2, 0, 1, 3).reshape(s, b * kv_head, head_dim)
        v_cache = v_base.permute(2, 0, 1, 3).reshape(s, b * kv_head, head_dim)

        if compress_cache:
            k_cache = self.compressed_device.compress(k_cache, comp_config)
            v_cache = self.compressed_device.compress(v_cache, comp_config)
        else:
            k_cache = TorchTensor.create_from_torch(k_cache, self)
            v_cache = TorchTensor.create_from_torch(v_cache, self)

        return TorchTensor.create_from_torch(value, self), k_cache, v_cache, w_q, w_k, partial_weight_index


    def mha_gen_qwen_infin(self, inputs, attention_mask, w_q, w_k, w_v,
                           w_out, w_ln, q_ln, k_ln, n_head,
                           k_cache, v_cache, donate,
                           attn_sparsity, compress_cache, comp_config,
                           eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings,
                           p_w_q, partial_k_cache, speculation_stream, alpha, max_num_kv, attn_topk=None):
        """Decoding-phase MHA with prefetch_idx generation via speculative attention."""
        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        scaling = head_dim ** -0.5
        cache_len = k_cache.data.shape[0]

        # Decompress weights
        
        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)

        # --- Speculative Attention: Generate prefetch_idx ---
        prefetch_idx = None
        if p_w_q is not None:
            with torch.cuda.stream(speculation_stream):
                prefetch_idx = speculate_attention(hidden, p_w_q, partial_k_cache, n_head, alpha, max_num_kv)

        # --- Normal Attention ---
        q = F.linear(hidden, w_q)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, num_key_value_heads, head_dim)

        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm_gqa(k, weight=k_ln, eps=eps)

        q = q.transpose(1, 2)  # (b, n_head, 1, d)
        k = k.transpose(1, 2)  # (b, n_kv, 1, d)
        v = v.transpose(1, 2)  # (b, n_kv, 1, d)

        # RoPE
        position_id = src_s - 1
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=inputs.device, dtype=inputs.dtype) / head_dim))
        t = torch.tensor([position_id], device=inputs.device, dtype=inputs.dtype)
        freqs = torch.outer(t, freqs).repeat_interleave(2, dim=-1)
        cos, sin = freqs.cos(), freqs.sin()
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # GQA
        # k_new = repeat_kv(k, num_key_value_groups)  # (b, n_head, 1, d) 
        k_new = k
        v_new = repeat_kv(v, num_key_value_groups) # only repeat v because of infinigen

        # Reshape new k/v for cache update: (1, b * n_head, d)
        k_cache_new = k_new.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        v_cache_new = v_new.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        use_sparse = (attn_topk is not None) or (attn_sparsity < 1.0)

        # Update full cache
        if isinstance(k_cache, TorchTensor):
            if not use_sparse:
                if compress_cache:
                    k_all = k_cache.device.decompress(k_cache)[:cache_len]
                    v_all = v_cache.device.decompress(v_cache)[:cache_len]
                else:
                    k_all = k_cache.data[:cache_len]
                    v_all = v_cache.data[:cache_len]

                k_all = torch.cat([k_all, k_cache_new], dim=0)  # (cache_len+1, b * n_head, d)
                v_all = torch.cat([v_all, v_cache_new], dim=0)

                # Reshape for bmm: (b * n_head, d, s), (b * n_head, s, d)
                k_all = k_all.permute(1, 2, 0)  # (b*n_head, d, cache_len+1)
                v_all = v_all.permute(1, 0, 2)  # (b*n_head, cache_len+1, d)

                q_t = q.transpose(1, 2).reshape(b * n_head, tgt_s, head_dim)  # (b*n_head, 1, d)

                attn_scores = torch.bmm(q_t, k_all) / (head_dim ** 0.5)
                attn_probs = F.softmax(attn_scores, dim=-1)
                value = torch.bmm(attn_probs, v_all)  # (b*n_head, 1, d)

                value = value.view(b, n_head, tgt_s, head_dim).transpose(1, 2).reshape(b, tgt_s, n_head*head_dim)
            else:
                # Sparse attention with fixed top-k (or ratio) over history tokens.
                # If we have a full cache buffer (>= src_s), slice to src_s and update the last position.
                # If we only have a prefetched sparse cache (< src_s), build a local attention window by appending
                # the current token, and use an all-True local mask.
                if compress_cache:
                    k_base = k_cache.device.decompress(k_cache)
                    v_base = v_cache.device.decompress(v_cache)
                else:
                    k_base = k_cache.data
                    v_base = v_cache.data

                if k_base.shape[0] >= src_s:
                    k_hist = k_base[:src_s]
                    v_hist = v_base[:src_s]
                    k_hist[src_s - 1:src_s] = k_cache_new
                    v_hist[src_s - 1:src_s] = v_cache_new
                    local_s = src_s
                    mask = attention_mask.data[:, :local_s].view(b, 1, 1, local_s)
                else:
                    k_hist = torch.cat([k_base, k_cache_new], dim=0)
                    v_hist = torch.cat([v_base, v_cache_new], dim=0)
                    local_s = k_hist.shape[0]
                    mask = torch.ones((b, 1, 1, local_s), dtype=torch.bool, device=inputs.device)

                k_all = k_hist.view(local_s, b, n_head, head_dim).permute(1, 2, 0, 3)  # (b,n_head,local_s,d)
                v_all = v_hist.view(local_s, b, n_head, head_dim).permute(1, 2, 0, 3)

                # scores/probs in fp32 for stability
                scores = torch.matmul(
                    q.to(torch.float32),
                    k_all.transpose(-1, -2).to(torch.float32),
                ) / (head_dim ** 0.5)  # (b,n_head,1,local_s)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                probs = torch.softmax(scores, dim=-1)  # (b,n_head,1,local_s)

                hist_probs = probs[..., :-1]
                max_history = hist_probs.shape[-1]
                if max_history <= 0:
                    weight_last = probs[..., -1:]  # (b,n_head,1,1)
                    attn_out = v_all[:, :, -1:, :].to(torch.float32) * weight_last.to(torch.float32)
                else:
                    if attn_topk is not None:
                        topk = max(1, min(int(attn_topk), max_history))
                    else:
                        topk = int(attn_sparsity * max_history)
                        topk = max(1, min(topk, max_history))

                    topk_weights, topk_indices = torch.topk(hist_probs, k=topk, dim=-1, sorted=False)  # (b,n_head,1,topk)
                    v_hist = v_all[:, :, :-1, :]  # (b,n_head,src_s-1,d)
                    gather_idx = topk_indices.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, head_dim)  # (b,n_head,topk,d)
                    v_sel = torch.gather(v_hist, dim=2, index=gather_idx)  # (b,n_head,topk,d)
                    v_sel = torch.cat([v_sel, v_all[:, :, -1:, :]], dim=2)  # (b,n_head,topk+1,d)

                    weights = torch.cat(
                        [topk_weights.squeeze(2), probs[..., -1:].squeeze(2)],
                        dim=-1,
                    )  # (b,n_head,topk+1)
                    attn_out = torch.sum(v_sel.to(torch.float32) * weights.to(torch.float32).unsqueeze(-1), dim=2).unsqueeze(2)  # (b,n_head,1,d)

                value = attn_out.to(inputs.data.dtype).view(b, n_head, tgt_s, head_dim).transpose(1, 2).reshape(b, tgt_s, n_head * head_dim)
        else:
            raise NotImplementedError("Mixed device attention not supported")

        # Final output
        value = F.linear(value, w_out)
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # Create new cache entry
        if compress_cache:
            k_cache_new = self.compressed_device.compress(k_cache_new, comp_config)
            v_cache_new = self.compressed_device.compress(v_cache_new, comp_config)
        else:
            k_cache_new = TorchTensor.create_from_torch(k_cache_new, self)
            v_cache_new = TorchTensor.create_from_torch(v_cache_new, self)

        return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new, prefetch_idx
    
    
    ########################################################
    
    
    ######################################################## qwen mha for infinigen2
    def mha_qwen_infin_v2(self, inputs, attention_mask, w_q, w_k, w_v,
                      w_out, w_ln, q_ln, k_ln, n_head, donate, compress_cache, comp_config,
                      eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings,
                      warmup, partial_weight_ratio):
        """Multi-head attention for Qwen3 with InfiniGen's warmup, skew, and column-wise partial cache support."""
        b, s, h = inputs.shape
        scaling = head_dim ** -0.5

        # Decompress weights if compressed

        # Get position embeddings
        if position_embeddings is not None:
            cos_global, sin_global = position_embeddings
            if cos_global.shape[-1] != head_dim:
                cos, sin = get_rotary_position_embeddings(s, head_dim, cos_global.device, inputs.dtype)
            else:
                cos, sin = cos_global[:s], sin_global[:s]
        else:
            cos, sin = get_rotary_position_embeddings(s, head_dim, inputs.device, inputs.dtype)

        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)

        # Linear projections
        q = F.linear(hidden, w_q)  # (b, s, h)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        print(f"[backend debug] w_k shape = {w_k.shape}, w_v shape = {w_v.shape}", flush=True)
        
        # judge the k cache size
        is_mha = False
        if w_k.shape[0] == n_head*head_dim:
            is_mha = True
        
        # Partial weight index: column-wise selection (d' = int(head_dim * ratio))
        partial_weight_index = None
        if (not warmup) and (partial_weight_ratio is not None):
            partial_weight_index = partial_weight_index_generation(q, n_head, head_dim, partial_weight_ratio)

        # Reshape: (b, s, h) -> (b, s, n_head/n_kv, head_dim)
        q = q.view(b, s, n_head, head_dim)
        kv_head = n_head if is_mha else num_key_value_heads
        k = k.view(b, s, kv_head, head_dim)
        v = v.view(b, s, kv_head, head_dim)

        # Q/K Norm
        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm(k, weight=k_ln, eps=eps)

        # Transpose: (b, s, n_head/n_kv, d) -> (b, n_head/n_kv, s, d)
        q = q.transpose(1, 2)  # (b, n_head, s, d)
        k = k.transpose(1, 2)  # (b, n_kv, s, d)
        v = v.transpose(1, 2)  # (b, n_kv, s, d)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # GQA: repeat k/v for attention only, keep base for cache
        k_base = k
        v_base = v
        if not is_mha:
            k = repeat_kv(k_base, num_key_value_groups)
            v = repeat_kv(v_base, num_key_value_groups)
        else:
            k = k_base
            v = v_base

        # Skew mechanism during warmup
        if warmup:
            print(f"[backend debug] skew_gqa before w_q shape = {w_q.shape}, w_k shape = {w_k.shape}")
            w_q.data, w_k.data = skew_gqa(q, k, w_q, w_k, n_head, num_key_value_heads, head_dim)
            print(f"[backend debug] skew_gqa after w_q shape = {w_q.shape}, w_k shape = {w_k.shape}")
        

        # Causal + padding mask only when padding exists to reduce peak memory.
        has_padding = attention_mask is not None and not bool(attention_mask.all().item())
        if has_padding:
            causal_mask = torch.triu(
                torch.ones((s, s), device=q.device, dtype=torch.bool),
                diagonal=1
            )
            mask = attention_mask.view(b, 1, 1, s)
            combined_mask = mask & ~causal_mask
            attn_mask = torch.zeros((b, 1, s, s), dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(~combined_mask, torch.finfo(q.dtype).min)
            value = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,  # 手动构建了 mask
                scale=None        # 如果 q 已缩放
            )
        else:
            value = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=None
            )
        

        # Output
        # value = torch.matmul(attn_weights, v)  # (b, n_head, s, d)
        # value = value.transpose(1, 2).reshape(b, s, h)  # (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, n_head*head_dim)  # (b, s, n_head*d)
        
        
        value = F.linear(value, w_out)
        value.add_(inputs.data)  # Residual

        # Donate
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # KV Cache: (s, b * n_kv_head, head_dim)
        k_cache = k_base.permute(2, 0, 1, 3).reshape(s, b * kv_head, head_dim)
        v_cache = v_base.permute(2, 0, 1, 3).reshape(s, b * kv_head, head_dim)

        if compress_cache:
            k_cache = self.compressed_device.compress(k_cache, comp_config)
            v_cache = self.compressed_device.compress(v_cache, comp_config)
        else:
            k_cache = TorchTensor.create_from_torch(k_cache, self)
            v_cache = TorchTensor.create_from_torch(v_cache, self)

        return TorchTensor.create_from_torch(value, self), k_cache, v_cache, w_q, w_k, partial_weight_index


    def mha_gen_qwen_infin_v2(self, inputs, attention_mask, w_q, w_k, w_v,
                              w_out, w_ln, q_ln, k_ln, n_head,
                              k_cache, v_cache, donate,
                              attn_sparsity, compress_cache, comp_config,
                              eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings,
                              p_w_q, partial_k_cache, speculation_stream, alpha, max_num_kv, attn_topk=None,
                              prefetch_grouped=False):
        """Decoding-phase MHA with prefetch_idx generation via speculative attention."""
        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        scaling = head_dim ** -0.5
        cache_len = k_cache.data.shape[0]

        # Decompress weights
        
        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)

        # --- Speculative Attention: Generate prefetch_idx ---
        prefetch_idx = None
        pad_idx = None
        if p_w_q is not None:
            with torch.cuda.stream(speculation_stream):
                if prefetch_grouped:
                    prefetch_idx, pad_idx = speculate_attention_grouped(
                        hidden, p_w_q, partial_k_cache, n_head,
                        num_key_value_heads, max_num_kv
                    )
                else:
                    prefetch_idx = speculate_attention(
                        hidden, p_w_q, partial_k_cache, n_head, alpha, max_num_kv
                    )

        # --- Normal Attention ---
        q = F.linear(hidden, w_q)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        is_mha = (w_k.shape[0] == n_head * head_dim)
        kv_head = n_head if is_mha else num_key_value_heads
        rep = n_head // kv_head

        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, kv_head, head_dim)
        v = v.view(b, tgt_s, kv_head, head_dim)

        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm_gqa(k, weight=k_ln, eps=eps)

        q = q.transpose(1, 2)  # (b, n_head, 1, d)
        k = k.transpose(1, 2)  # (b, n_kv, 1, d)
        v = v.transpose(1, 2)  # (b, n_kv, 1, d)

        # RoPE
        position_id = src_s - 1
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=inputs.device, dtype=inputs.dtype) / head_dim))
        t = torch.tensor([position_id], device=inputs.device, dtype=inputs.dtype)
        freqs = torch.outer(t, freqs).repeat_interleave(2, dim=-1)
        cos, sin = freqs.cos(), freqs.sin()
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # GQA: repeat for attention only, keep base for cache
        k_base = k
        v_base = v

        # Reshape new k/v for cache update: (1, b * n_kv, d)
        k_cache_new = k_base.permute(1, 0, 2, 3).reshape(tgt_s, b * kv_head, head_dim)
        v_cache_new = v_base.permute(1, 0, 2, 3).reshape(tgt_s, b * kv_head, head_dim)

        use_sparse = (attn_topk is not None) or (attn_sparsity < 1.0)

        # Update full cache
        if isinstance(k_cache, TorchTensor):
            if not use_sparse:
                if compress_cache:
                    k_all = k_cache.device.decompress(k_cache)[:cache_len]
                    v_all = v_cache.device.decompress(v_cache)[:cache_len]
                else:
                    k_all = k_cache.data[:cache_len]
                    v_all = v_cache.data[:cache_len]

                k_all = torch.cat([k_all, k_cache_new], dim=0)  # (cache_len+1, b * n_kv, d)
                v_all = torch.cat([v_all, v_cache_new], dim=0)

                if rep > 1:
                    k_all = repeat_kv_cache(k_all, rep, kv_head)
                    v_all = repeat_kv_cache(v_all, rep, kv_head)

                k_all = k_all.permute(1, 0, 2)  # (b*n_head, cache_len+1, d)
                v_all = v_all.permute(1, 0, 2)

                k_all = k_all.view(b, n_head, cache_len+1, head_dim)
                v_all = v_all.view(b, n_head, cache_len+1, head_dim)

                print(f"q shape = {q.shape}", flush=True)
                print(f"k_all shape = {k_all.shape}", flush=True)
                print(f"v_all shape = {v_all.shape}", flush=True)

                value = F.scaled_dot_product_attention(
                    q, k_all, v_all,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=None
                )  # (b, n_head, 1, d)


                value = value.view(b, n_head, tgt_s, head_dim).transpose(1, 2).reshape(b, tgt_s, n_head*head_dim)
            else:
                # Sparse attention with fixed top-k (or ratio) over history tokens.
                # If we have a full cache buffer (>= src_s), slice to src_s and update the last position.
                # If we only have a prefetched sparse cache (< src_s), build a local attention window by appending
                # the current token, and use an all-True local mask.
                if compress_cache:
                    k_base = k_cache.device.decompress(k_cache)
                    v_base = v_cache.device.decompress(v_cache)
                else:
                    k_base = k_cache.data
                    v_base = v_cache.data

                if k_base.shape[0] >= src_s:
                    k_hist = k_base[:src_s]
                    v_hist = v_base[:src_s]
                    k_hist[src_s - 1:src_s] = k_cache_new
                    v_hist[src_s - 1:src_s] = v_cache_new
                    local_s = src_s
                    mask = attention_mask.data[:, :local_s].view(b, 1, 1, local_s)
                else:
                    k_hist = torch.cat([k_base, k_cache_new], dim=0)
                    v_hist = torch.cat([v_base, v_cache_new], dim=0)
                    local_s = k_hist.shape[0]
                    mask = torch.ones((b, 1, 1, local_s), dtype=torch.bool, device=inputs.device)

                if rep > 1:
                    k_hist = repeat_kv_cache(k_hist, rep, kv_head)
                    v_hist = repeat_kv_cache(v_hist, rep, kv_head)

                k_all = k_hist.view(local_s, b, n_head, head_dim).permute(1, 2, 0, 3)  # (b,n_head,local_s,d)
                v_all = v_hist.view(local_s, b, n_head, head_dim).permute(1, 2, 0, 3)

                scores = torch.matmul(q.to(torch.float32), k_all.transpose(-1, -2).to(torch.float32)) / (head_dim ** 0.5)  # (b,n_head,1,local_s)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                probs = torch.softmax(scores, dim=-1)

                hist_probs = probs[..., :-1]
                max_history = hist_probs.shape[-1]
                if max_history <= 0:
                    weight_last = probs[..., -1:]  # (b,n_head,1,1)
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

                value = attn_out.to(inputs.data.dtype).view(b, n_head, tgt_s, head_dim).transpose(1, 2).reshape(b, tgt_s, n_head*head_dim)
        else:
            raise NotImplementedError("Mixed device attention not supported")

        # Final output
        value = F.linear(value, w_out)
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # Create new cache entry
        if compress_cache:
            k_cache_new = self.compressed_device.compress(k_cache_new, comp_config)
            v_cache_new = self.compressed_device.compress(v_cache_new, comp_config)
        else:
            k_cache_new = TorchTensor.create_from_torch(k_cache_new, self)
            v_cache_new = TorchTensor.create_from_torch(v_cache_new, self)

        if prefetch_grouped:
            if prefetch_idx is None:
                return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new, None
            return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new, (prefetch_idx, pad_idx)
        return TorchTensor.create_from_torch(value, self), k_cache_new, v_cache_new, prefetch_idx
    
    ########################################################
    
    
    ######################################################## quest without prefetch
    
    def mha_qwen_quest(self, inputs, attention_mask, w_q, w_k, w_v,
             w_out, w_ln, q_ln, k_ln, n_head, donate, compress_cache, comp_config,
             eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings):
        """Multi-head attention for Qwen3 (prefill phase)."""
        b, s, h = inputs.shape
        scaling = head_dim ** -0.5

        # Get position embeddings
        if position_embeddings is not None:
            cos_global, sin_global = position_embeddings
            if cos_global.shape[-1] != head_dim:
                cos, sin = get_rotary_position_embeddings(s, head_dim, cos_global.device, inputs.dtype)
            else:
                cos, sin = cos_global[:s], sin_global[:s]  # slice to current length
        else:
            cos, sin = get_rotary_position_embeddings(s, head_dim, inputs.device, inputs.dtype)
            
        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)

        # Linear projections
        # hidden = inputs.data
        q = F.linear(hidden, w_q)  # (b, s, h)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        # Reshape: (b, s, h) -> (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, num_key_value_heads, head_dim)
        v = v.view(b, s, num_key_value_heads, head_dim)

        # Apply Q/K Norm
        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm(k, weight=k_ln, eps=eps)

        # ✅ 转置为 (b, n_head, s, head_dim)
        q = q.transpose(1, 2)  # → (b, n_head, s, head_dim)
        k = k.transpose(1, 2)  # → (b, n_kv, s, head_dim)
        v = v.transpose(1, 2)  # → (b, n_kv, s, head_dim)
        
        
        # ✅ 使用 HF 的 apply_rotary_pos_emb (unsqueeze_dim=1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)  # (b, n_head, s, d)
        
        # ✅ GQA: repeat k/v after RoPE
        k_before_repeat = k
        v_before_repeat = v
        k = repeat_kv(k, num_key_value_groups)  # (b, n_head, s, d) -> (b, n_head*rep, s, d)
        v = repeat_kv(v, num_key_value_groups)

        # print(f"mha after q shape = {q.shape} dtype = {q.dtype}", flush=True)
        # print(f"mha after k shape = {k.shape} dtype = {k.dtype}", flush=True)
        
        # Memory-efficient attention (avoids materializing (b, n_head, s, s))
        if attention_mask is not None and not bool(attention_mask.data.all().item()):
            raise NotImplementedError(
                "Prefill attention with padding mask is not supported in this backend; "
                "please provide non-padded inputs or implement a proper attn_mask."
            )
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        value = value.transpose(1, 2).reshape(b, s, n_head*head_dim)  # (b, s, n_head*head_dim)
        value = F.linear(value, w_out)
        value.add_(inputs.data)  # Residual

        # Donate
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # ✅ Prepare KV Cache: 保存原始 k/v (before repeat)
        # k: (b, n_kv, s, d) -> (s, b * n_kv, d)
        k_cache = k_before_repeat.transpose(0, 2).transpose(1, 3).reshape(s, b * num_key_value_heads, head_dim)
        v_cache = v_before_repeat.transpose(0, 2).transpose(1, 3).reshape(s, b * num_key_value_heads, head_dim)

        if compress_cache:
            k_cache = self.compressed_device.compress(k_cache, comp_config)
            v_cache = self.compressed_device.compress(v_cache, comp_config)
        else:
            k_cache = TorchTensor.create_from_torch(k_cache, self)
            v_cache = TorchTensor.create_from_torch(v_cache, self)

        return TorchTensor.create_from_torch(value, self), k_cache, v_cache
    
    def mha_gen_qwen_quest(self, inputs, attention_mask, w_q, w_k, w_v,
                 w_out, w_ln, q_ln, k_ln, n_head,
                 quest_kv_manager, sparse_rate, donate,
                 attn_sparsity, compress_cache, comp_config,
                 eps, head_dim, num_key_value_groups, num_key_value_heads, position_embeddings):
        """Multi-head attention for Qwen3 (decoding phase)."""
        b, tgt_s, h = inputs.shape
        d = head_dim
        src_s = attention_mask.shape[1]
        scaling = head_dim ** -0.5
        
        # RMS norm
        hidden = rms_norm(inputs.data, weight=w_ln, eps=eps)
        
        # Project
        # hidden = inputs.data
        q = F.linear(hidden, w_q)  # (b, 1, h)
        k = F.linear(hidden, w_k)
        v = F.linear(hidden, w_v)

        # Reshape
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, num_key_value_heads, head_dim)
        v = v.view(b, tgt_s, num_key_value_heads, head_dim)

        # Q/K Norm
        q = rms_norm(q, weight=q_ln, eps=eps) * scaling
        k = rms_norm(k, weight=k_ln, eps=eps)

        # Transpose
        q = q.transpose(1, 2)  # (b, n_head, 1, head_dim)
        k = k.transpose(1, 2)  # (b, n_kv, 1, head_dim)
        v = v.transpose(1, 2)  # (b, n_kv, 1, head_dim)
        
        
        # print(f"first k = {k.shape}", flush=True)
        # print(f"first v = {v.shape}", flush=True)

        # RoPE
        # get position embedding
        position_id = src_s - 1  # int
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=inputs.device, dtype=inputs.dtype) / head_dim))
        t = torch.tensor([position_id], device=inputs.device, dtype=inputs.dtype)
        freqs = torch.outer(t, freqs)  # (1, head_dim//2)
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # (1, head_dim)
        sin = freqs.sin().repeat_interleave(2, dim=-1)  # (1, head_dim)
        
        # cos, sin = position_embeddings
        # cos = cos[src_s - 1:src_s]  # 只取最后一个位置
        # sin = sin[src_s - 1:src_s]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # GQA
        k_new = k # (b, n_head, s, d)
        v_new = v

        # Current k/v
        k_new = k.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)
        v_new = v.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)
        
        ###################################### Get sparse KV cache
        # sparse_page_id = quest_kv_manager.
        top_page_num = int((src_s * sparse_rate) // quest_kv_manager.page_size)
        sparse_pages_idx = quest_kv_manager.quest_select_for_batch(q, top_k=top_page_num)
        sparse_k, sparse_v = quest_kv_manager.gather_sparse_kv_to_gpu(sparse_pages_idx)
        
        print(f"sparse_k shape = {sparse_k.shape}", flush=True)
        assert sparse_k.shape[1] == (b * num_key_value_heads), "sparse_k shape is error"
        
        k_all = torch.concat((sparse_k, k_new), dim=0)
        v_all = torch.concat((sparse_v, v_new), dim=0)
        
        ######################################

        # Handle cache
        if attn_sparsity >= 1.0:
#                 if compress_cache:
#                     k_all = k_cache.device.decompress(k_cache)[:src_s]  # (s, b * n_kv, d)
#                     v_all = v_cache.device.decompress(v_cache)[:src_s]
#                 else:
#                     k_all = k_cache.data[:src_s]  # (s, b * n_kv, d)
#                     v_all = v_cache.data[:src_s]

#                 # 更新最后一个 token
#                 k_all[src_s - 1:src_s] = k_new  # k_new: (1, b * n_kv, d)
#                 v_all[src_s - 1:src_s] = v_new

            # ✅ GQA: repeat after update
            k_all = repeat_kv_cache(k_all, num_key_value_groups, num_key_value_heads)  # (s, b * n_head, d)
            v_all = repeat_kv_cache(v_all, num_key_value_groups, num_key_value_heads)

            # ✅ 转置为 bmm 友好格式
            k_all = k_all.permute(1, 2, 0)  # (b * n_head, d, s)
            v_all = v_all.permute(1, 0, 2)  # (b * n_head, s, d)

            # Attention
            q_t = q.transpose(1, 2)  # (b, 1, n_head, d) -> (b, n_head, 1, d)
            q_t = q_t.reshape(b * n_head, 1, head_dim)  # (b * n_head, 1, d)

            attn_scores = torch.bmm(q_t, k_all) / (head_dim ** 0.5)  # (b * n_head, 1, s)
            attn_probs = F.softmax(attn_scores, dim=-1)  # (b * n_head, 1, s)
            attn_output = torch.bmm(attn_probs, v_all)  # (b * n_head, 1, d)

            attn_output = attn_output.view(b, n_head, 1, d).transpose(1, 2).reshape(b, 1, n_head*head_dim)

        # Reshape output
        # attn_output = attn_output.transpose(1, 2).view(b, tgt_s, h)
        attn_output = F.linear(attn_output, w_out)
        attn_output.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # Create new cache entries (original kv, not repeated)
        # k_new = k_new.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)
        # v_new = v_new.transpose(0, 2).transpose(1, 3).reshape(tgt_s, b * num_key_value_heads, head_dim)

        if compress_cache:
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(attn_output, self), k_new, v_new
    
    
    
    
    ########################################################
    
    
    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity, attn_topk=None):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)

        # decide how many historical tokens to keep (exclude the latest position)
        max_history = attn_weights.shape[2] - 1
        if max_history <= 0:
            # no history to sparsify; fall back to dense attention value
            return self._attention_value(q, k, v_new, mask, b, src_s, tgt_s, n_head, head_dim)

        if attn_topk is not None:
            topk = max(1, min(int(attn_topk), max_history))
        else:
            topk = int(attn_sparsity * max_history)
            topk = max(1, min(topk, max_history))

        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), torch.bfloat16)
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

    def mlp(self, inputs, wi, w_gate, wo, w_ln, donate, eps, act_fn):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        gate_out = F.linear(out, w_gate.data)

        out = F.linear(out, wi.data)
        # F.relu(out, inplace=True)

        out = act_fn(gate_out) * out

        out = F.linear(out, wo.data)

        # print(out[:,:,:6])

        out.add_(inputs.data) # Residual
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
        for t in self.copy_threads:
            t.start()

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
                    "Use float16/float32 or disable disk offloading for bfloat16."
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
        num_key_value_heads, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_key_value_heads, config.hidden_size, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_key_value_heads, config.head_dim)
        k_cache = self.allocate(shape, torch.bfloat16)
        v_cache = self.allocate(shape, torch.bfloat16)
        return k_cache, v_cache

    def init_cache_one_gpu_batch_infin(self, config, task, policy):
        num_key_value_heads, prompt_len, gen_len, gpu_batch_size = (
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            task.prompt_len, task.gen_len, policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_key_value_heads, config.head_dim)
        k_cache = self.allocate(shape, torch.bfloat16)
        v_cache = self.allocate(shape, torch.bfloat16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
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
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            torch_dtype = np_dtype_to_torch_dtype[dtype]

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
        k_cache = self.allocate(shape, torch.bfloat16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, torch.bfloat16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache

    def init_cache_one_gpu_batch_infin(self, config, task, policy):
        num_kv_head, prompt_len, gen_len, gpu_batch_size = (
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            task.prompt_len, task.gen_len, policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_kv_head, config.head_dim)

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
        k_cache = self.allocate(shape, torch.bfloat16, seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, torch.bfloat16, seg_lengths=lens, pin_memory=pin_memory)
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

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
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
