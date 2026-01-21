import torch
import time
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import pynvml

import sys

# i am update

def print_gpu_memory():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count):
        torch.cuda.empty_cache()
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"GPU {i}: "
              f"Used: {mem_info.used / 1024**2:.2f} MB, "
              f"Total: {mem_info.total / 1024**2:.2f} MB, "
              f"Free: {mem_info.free / 1024**2:.2f} MB", flush=True)
    
    pynvml.nvmlShutdown()

class CacheManager:
    '''
        Build to manage the cache for all layers.
    '''
    
    def __init__(self, 
                 basic_group_head_ids, 
                 layer_head_num, 
                 batch_size, 
                 head_num, 
                 max_sparse_len, 
                 head_dim,
                 dtype,
                 gpu_cache_pred=1,
                 cpu_cache_pred=1,
                ):
        """
        初始化 CacheManager，使用字典保存不同 layer_id 对应的缓存数据。
        remove update pred
        """
        self._caches = {}  # key: layer_id, value: cache data
        self._update_recode = {}

        # tmp change 
        update_pred = 4
        self._update_pred = update_pred
        
        self._basic_group_head_ids = basic_group_head_ids # key: layer_id, value: layer group head ids
        self._cur_group_ids = {} # e.g. {1:[[1,2],[3,4],[5,6]]}

        # print("init self._basic_group_head_ids = ", self._basic_group_head_ids)
        self.layer_head_number = layer_head_num
        self.executor = ThreadPoolExecutor()  # 可复用线程池

        self._layer_tmp_key = None
        self._layer_tmp_value = None

        # unhit kv pinned memory
        self._pinned_unhit_k_list = []
        self._pinned_unhit_v_list = []
        self._pinned_n_p = 0

        # pinned space for cached tensor
        self._pinned_cached_k_list = {}
        self._pinned_cached_v_list = {}
        self.gpu_cache_pred = 1
        self.cpu_cache_pred = 1
        self._max_cpu_cached_len = int(max_sparse_len * self.cpu_cache_pred)
        self._max_gpu_cached_len = int(max_sparse_len * self.gpu_cache_pred)
        # self._max_cached_len = 0
        # self._pinned_cache_shape = {}

        # gpu cache
        self._use_gpu_cache = {} # should be initial in create
        self._gpu_cached_group_kv = {}
        
        # cpu cache
        self._cpu_cached_group_kv = {}

        # data ana
        self.cnt = 0

        # update according to max cache len
        self._require_update = {}
        self.max_unhit_rate = 0.4 # max 40% unhit kv
        
        # build unhit k space and unhit v space
        # share acrss all layers
        self.bh = batch_size*head_num
        self.max_sparse_len = max_sparse_len
        self.head_dim = head_dim
        self.dtype = dtype
        
        # print(f"shape = {(self.max_sparse_len, self.bh, self.head_dim)}", flush=True)
        # print(f"dtype = {self.dtype}", flush=True)
        
        tmp_key = torch.empty((self.max_sparse_len, self.bh, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
        tmp_value = torch.empty((self.max_sparse_len, self.bh, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
        self._pinned_unhit_k_list.append(tmp_key)
        self._pinned_unhit_v_list.append(tmp_value)
        
        # gpu cache map version
        self.max_token_len = 0 # initial
        self.layer_cache_maps = {}
        self.layer_unhit_maps = {}
        self.layer_unhit_idxs = {}
        self.layer_unhit_lengths = {}
        self.layer_sparse_maps = {}
        
        # flat space version
        # flat_unhit_shape = ((self.max_sparse_len+1) * self.bh, self.head_dim)
        # self._pinned_flat_unhit_k = torch.empty(flat_unhit_shape, dtype=self.dtype, device='cpu', pin_memory=True)
        # self._pinned_flat_unhit_v = torch.empty(flat_unhit_shape, dtype=self.dtype, device='cpu', pin_memory=True)
            
    

    def init_basic_group_head_ids(self, layer_group_head_ids):
        for l in range(len(layer_group_head_ids)):
            self._basic_group_head_ids[l] = layer_group_head_ids[l]    

    def add_cache(self, device, layer_id, batch_size, head_num, max_token_len, sparse_len, hidden_size, use_gpu_cache=True):
        """
        添加或更新指定 layer_id 的缓存数据。
        
        :param layer_id: 缓存标识符（字符串）
        """
        if layer_id in self._caches:
            raise ValueError("Python ERROR! [add_cache] layer id is used!")
        
        if not use_gpu_cache:
            raise ValueError("Python ERROR! [add_cache] stawman only use gpu cache!")

        # print(f"CacheManager add_cache: Layer#{layer_id} max_token_len={max_token_len}", flush=True)
        
        self._require_update[layer_id] = True # 第一次load必须更新
        self.max_token_len = max(self.max_token_len, max_token_len)

        # 在这里决定他是否要使用gpu cache
        self._use_gpu_cache[layer_id] = use_gpu_cache
        
        pinned_cache_shape = (self.max_sparse_len, self.bh, self.head_dim)
        k_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
        v_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
        self._pinned_cached_k_list[layer_id] = [k_cache_space]
        self._pinned_cached_v_list[layer_id] = [v_cache_space]

        return 0

    def remove_cache(self, layer_id):
        """
        移除指定 layer_id 的缓存。
        
        :param layer_id: 缓存标识符
        """
        if layer_id in self._caches:
            del self._caches[layer_id]

    def has_cache(self, layer_id) -> bool:
        """
        检查是否存在指定 layer_id 的缓存。
        
        :param layer_id: 缓存标识符
        :return: 布尔值
        """
        return layer_id in self._caches

    def clear_caches(self):
        """
        清空所有缓存。
        """
        self._caches.clear()

    def cache_count(self) -> int:
        """
        返回当前缓存的数量。
        
        :return: 整数
        """
        return len(self.layer_cache_maps)

    # def get_layer_group_id(self, layer_id):
    #     return self._cur_group_ids[layer_id]
    
    def get_layer_cache(self, layer_id):
        return self._caches[layer_id]
    
    def cache_miss_check(self, layer_id, prefetch_idx):
        # if require update, directly return prefetch_idx
        if self._require_update[layer_id]:
            # print(f"cache_miss_check: Layer#{layer_id} here?", flush=True)
            return prefetch_idx
        
        
        # print(f"cache_miss_check: Layer#{layer_id} reach illegal", flush=True)
        assert self.max_token_len != 0, "Cachemanager cache_miss_check Error! max_token_len is Zero."
        
        W_cache = self.layer_cache_maps[layer_id]
        
        W_sparse_gpu = build_sparse_map(prefetch_idx, self.max_token_len, device='cuda')
        W_unhit_gpu = cache_miss_detection(W_cache, W_sparse_gpu)
        unhit_idx_gpu, lengths = unhit_map_to_padded_idx(W_unhit_gpu)
        unhit_idx_cpu = unhit_idx_gpu.cpu()
        
        
        print(f"cache_miss_check: Layer #{layer_id} unhit lengths = {lengths}", flush=True)
        
        self.layer_unhit_lengths[layer_id] = lengths
        self.layer_unhit_idxs[layer_id] = unhit_idx_cpu
        self.layer_sparse_maps[layer_id] = W_sparse_gpu
        
        # update cache map (only for gpu cache)
        # update_cache_map_(W_cache, W_sparse_gpu)
        # if self._use_gpu_cache[layer_id]:
        update_cache_map_(W_cache, W_sparse_gpu)
        
        return unhit_idx_cpu
        
    
    def gpu_cache_load(self, layer_id, transfer_stream, unhit_idx, pad_idx, all_k, all_v, dtype):
        '''
            prefetch_idx (n', 1, bh)
            pad_idx (1, 1, bh)
        '''

        # step0: get layer cache context
        cur_cache_map = self.layer_cache_maps[layer_id]
        cur_unhit_idx = unhit_idx.cpu()
        cur_pad_idx = pad_idx.cpu()
        cur_unhit_length = self.layer_unhit_lengths[layer_id]
        cur_sparse_maps = self.layer_sparse_maps[layer_id]
        
        group_cached_gpu_k, group_cached_gpu_v = self._gpu_cached_group_kv[layer_id]
        
        # step1: build unhit tensor (maybe we should use cpp module)
        bh = unhit_idx.shape[-1]
        max_unhit_len = unhit_idx.shape[0]
        
        
        # oor select: use cpu module to get 
        
        cur_pad_idx_list = cur_pad_idx[0][0].int().tolist()
        cur_unhit_length_list = cur_unhit_length.int().tolist()
        
        _C.select_kv_tensor_with_pad(
            cur_unhit_idx, 
            cur_unhit_length_list, 
            cur_pad_idx_list, 
            all_k, 
            all_v, 
            self._pinned_unhit_k_list[0],
            self._pinned_unhit_v_list[0]
        )
        
        # print("select_kv_tensor_with_pad success?", flush=True)
        
        
        # Step 2: manage as group
        tmp_unhit_k = self._pinned_unhit_k_list[0][:max_unhit_len, :, :]
        tmp_unhit_v = self._pinned_unhit_v_list[0][:max_unhit_len, :, :]
        group_unhit_k = [tmp_unhit_k]
        group_unhit_v = [tmp_unhit_v]
        
        
            
        # Step 3: unhit_k/v -> GPU 
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            # for i in range(len(group_cached_gpu_k)):
            group_final_k = [(group_cached_gpu_k[0], unhit_gpu_k[0])]
            group_final_v = [(group_cached_gpu_v[0], unhit_gpu_k[0])]
        
#         print("unhit_idx shape =", unhit_idx.shape)
#         group_cache_shape = [unhit.shape for unhit in group_cached_gpu_k]
#         print("group_cache_shape = ", group_cache_shape)
#         group_unhit_shape = [unhit.shape for unhit in group_unhit_k]
#         print("group_unhit_shape = ", group_unhit_shape)
        


        return group_final_k, group_final_v, None

 
    
    def load_and_update_gpu_cached(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype):


        # step 1: 如果预留的pinned tensor不够大, 创建新的pinned tensor
        if layer_id not in self._pinned_cached_k_list:
            pinned_cache_shape = (self.max_sparse_len, self.bh, self.head_dim)
            k_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, device=torch.device('cpu'), pin_memory=True)
            v_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, device=torch.device('cpu'), pin_memory=True)
            self._pinned_cached_k_list[layer_id] = [k_cache_space]
            self._pinned_cached_v_list[layer_id] = [v_cache_space]

        # Step 2: get pinned cache space
        layer_pinned_k_space = self._pinned_cached_k_list[layer_id][0]
        layer_pinned_v_space = self._pinned_cached_v_list[layer_id][0]
        cur_sparse_len = prefetch_idx.shape[0]
        final_cpu_k = [layer_pinned_k_space[:cur_sparse_len]]
        final_cpu_v = [layer_pinned_v_space[:cur_sparse_len]]
        

        # Step 3: 获取新的kv数据
        selected_k, selected_v = select_kv(prefetch_idx, all_k, all_v)
        group_cpu_k = [selected_k]
        group_cpu_v = [selected_v]
        

        # Step 4: 将数据拷贝到pinned_tensor中
        final_cpu_k[0].copy_(group_cpu_k[0])
        final_cpu_v[0].copy_(group_cpu_v[0])
        
        # Step 5: 发起 cache更新任务
        W_cache_map = build_sparse_map(prefetch_idx, self.max_token_len, 'cuda')
        self.layer_cache_maps[layer_id] = W_cache_map
        
        # Step 6: 并行开始 KV 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_gpu_k = [k.cuda(non_blocking=True) for k in final_cpu_k]
            group_gpu_v = [v.cuda(non_blocking=True) for v in final_cpu_v]


        # 更新 gpu cache
        if layer_id in self._gpu_cached_group_kv:
            del self._gpu_cached_group_kv[layer_id]
        self._gpu_cached_group_kv[layer_id] = (group_gpu_k, group_gpu_v)

        return group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v



    # 统一的更新和加载接口 （默认全是GPU Cache）
    def unified_load_api(self, layer_id, transfer_stream, unhit_idx, pad_idx, all_k, all_v, dtype):
        # 判断是否需要更新
        if self._require_update[layer_id]:
            # 直接更新cache
            self._update_recode[layer_id] = 0
            self._require_update[layer_id] = False
            
            group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_gpu_cached(layer_id, transfer_stream, unhit_idx, all_k, all_v, dtype)
            
            return (group_gpu_k, group_gpu_v, None)
        else:
            self._update_recode[layer_id] += 1
            
            # version 3
            group_final_k, group_final_v, unhit_id_map = self.gpu_cache_load(layer_id, transfer_stream, unhit_idx, pad_idx, all_k, all_v, dtype)
            
            if self._update_recode[layer_id] > self._update_pred:
                self._require_update[layer_id] = True
            
            return (group_final_k, group_final_v, unhit_id_map)
        
        
        # if self._use_gpu_cache[layer_id]:
        #     # 判断是否需要更新
        #     if self._require_update[layer_id]:
        #         # 直接更新cache
        #         self._update_recode[layer_id] = 0
        #         self._require_update[layer_id] = False
                
        #         group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_gpu_cached(layer_id, transfer_stream, unhit_idx, all_k, all_v, dtype)
                
        #         return (group_gpu_k, group_gpu_v, None)
        #     else:
        #         self._update_recode[layer_id] += 1
                
        #         # version 3
        #         group_final_k, group_final_v, unhit_id_map = self.gpu_cache_load(layer_id, transfer_stream, unhit_idx, pad_idx, all_k, all_v, dtype)
                
        #         # if self._update_recode[layer_id] > self._update_pred:
        #         #     self._require_update[layer_id] = True
                
        #         return (group_final_k, group_final_v, unhit_id_map)
        # else:
        #     # 判断是否需要更新
        #     if self._require_update[layer_id]:
        #         # 直接更新cache
        #         self._update_recode[layer_id] = 1
        #         self._require_update[layer_id] = False
        #         group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_cpu_cache(layer_id, transfer_stream, unhit_idx, all_k, all_v, dtype)
                
        #         return (group_gpu_k, group_gpu_v, None)
        #     else:
                
        #         self._update_recode[layer_id] += 1
                
        #         group_final_k, group_final_v, unhit_id_map = self.cpu_cache_load(layer_id, transfer_stream, unhit_idx, pad_idx, all_k, all_v, dtype)
                
        #         if self._update_recode[layer_id] > self._update_pred:
        #             self._require_update[layer_id] = True
                
        #         return (group_final_k, group_final_v, unhit_id_map)


        

##################################### 传统方法

def select_kv(prefetch_idx, k_cache, v_cache):
    """Selects and aggregates critical KV caches using speculated indices

    On the decoding stage, aggregates the critical KV caches corresponding to
    the speculated prefetch index using embedding function.

    Args:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)

    Returns:
        selected_k: selected key cache (n', bh, d)
        selected_v: selected value cache (n', bh, d)
    """

    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v


def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
    """Speculates the indices of the critical KV caches of next attention layer.

    On the decoding stage, by using the hidden states (layer i), partial query
    weight (layer i+1), and partial key cache (layer i+1), speculates the
    attention score of the next layer. After that, counts the number of
    critical tokens and gets the indcies of the top-k KV cache tokens with high
    attention scores.

    Args:
        hidden: Hidden states of layer i (b, 1, D)
        p_w_q: Partial query weight (D', D)
        p_k_c: Partial key cache (n, bh, d')

        Note that bh * d' == D'

    Returns:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
    """
    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
    max_ = torch.max(p_attn, dim=-1)[0]
    # thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    # count = torch.where(
    #     p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    # )
    # mean = torch.mean(torch.sum(count, dim=-1)).item()
    # prefetch_idx = torch.topk(
    #     p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
    # )[1]
    
    prefetch_idx = torch.topk(
        p_attn.permute(2, 1, 0), max_num_kv, dim=0
    )[1]

    return prefetch_idx


##################################### GPU Cache Map
# ========================================
# 核心 cache map 操作（GPU 上）
# ========================================

def build_sparse_map(sparse_ids, token_len, device='cuda'):
    """sparse_ids: (n', 1, BH) or (n', BH) → W_sparse: (BH, token_len)"""
    if sparse_ids.dim() == 3:
        sparse_ids = sparse_ids.squeeze(1)  # (n', BH)
    
    cur_sparse_ids = sparse_ids.to(device)
    BH = cur_sparse_ids.size(1)
    W_sparse = torch.zeros((BH, token_len), dtype=torch.bool, device=device)
    sparse_ids_T = cur_sparse_ids.t()  # (BH, n')
    W_sparse.scatter_(1, sparse_ids_T.long(), True)
    return W_sparse

def cache_miss_detection(W_cache, W_sparse):
    """GPU 上执行: ~W_cache & W_sparse"""
    return (~W_cache) & W_sparse

def update_cache_map_(W_cache, W_sparse):
    """原地更新: W_cache |= W_sparse"""
    W_cache.bitwise_or_(W_sparse)
    return W_cache


# ========================================
# ✅ 新增：在 GPU 上将 unhit_map 转为 padded (max_n, 1, BH) index
# ========================================

def unhit_map_to_padded_idx(unhit_map):
    """
    输入: unhit_map (BH, T), bool, on GPU
    输出: padded_idx (max_n, 1, BH), long, on GPU, padding 0
    """
    BH, T = unhit_map.shape
    device = unhit_map.device
    union_dtype = torch.int32

    # 生成每个 head 的 token indices
    arange = torch.arange(T, device=device, dtype=union_dtype).expand(BH, T)  # (BH, T)
    indices = torch.where(unhit_map, arange, torch.tensor(0, device=device))  # 未命中处填 0

    # 计算每个 head 的长度
    lengths = unhit_map.sum(dim=1)  # (BH,)
    max_n = lengths.max().item()
    if max_n == 0:
        return torch.zeros((0, 1, BH), dtype=torch.long, device=device), lengths

    # 排序：将有效值移到前面
    # 使用 stable sort 保证顺序
    _, sorted_indices = torch.sort(unhit_map.logical_not(), dim=1, stable=True)  # False 在前 → True 在前
    sorted_vals = torch.gather(indices, 1, sorted_indices)

    # 取前 max_n
    padded = sorted_vals[:, :max_n]  # (BH, max_n)
    padded_idx = padded.t().unsqueeze(1)  # (max_n, 1, BH)
    padded_idx = padded_idx.int()

    return padded_idx, lengths


##################################### Flat Communication

def pad_id_list_to_max(id_lists: List[List[int]], pad_value: int = 0, device='cuda') -> torch.Tensor:
    if not id_lists:
        return torch.empty(0, 0, dtype=torch.long)
    max_len = max(len(lst) for lst in id_lists)
    padded = [
        lst + [pad_value] * (max_len - len(lst)) for lst in id_lists
    ]
    return torch.tensor(padded, dtype=torch.long, device=device).t()


# wait for update as cpp module
def build_global_unhit_and_idmap_optimized(
    unhit_token_ids: List[List[int]],
    head_num: int,
    batch_size: int,
    head_dim: int,
    full_k_cpu: torch.Tensor,
    full_v_cpu: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    """
    向量化版本：避免 Python 循环，一次性提取所有未命中数据
    """
    BH = batch_size * head_num
    assert len(unhit_token_ids) == BH
    T = full_k_cpu.size(0)
    assert full_k_cpu.shape == (T, BH, head_dim)
    assert full_v_cpu.shape == full_k_cpu.shape

    all_token_ids = []
    all_head_ids = []
    id_lists = []
    current_offset = 1  # 0 保留给 padding

    for head_idx in range(BH):
        token_ids = unhit_token_ids[head_idx]
        if len(token_ids) > 0:
            all_token_ids.extend(token_ids)
            all_head_ids.extend([head_idx] * len(token_ids))
            ids = list(range(current_offset, current_offset + len(token_ids)))
            id_lists.append(ids)
            current_offset += len(token_ids)
        else:
            id_lists.append([])

    total_unhit = len(all_token_ids)
    if total_unhit == 0:
        return (torch.empty(0, head_dim, dtype=full_k_cpu.dtype),
                torch.empty(0, head_dim, dtype=full_v_cpu.dtype),
                id_lists)

    token_indices = torch.tensor(all_token_ids, dtype=torch.long)
    head_indices = torch.tensor(all_head_ids, dtype=torch.long)

    unhit_k_flat = full_k_cpu[token_indices, head_indices, :]
    unhit_v_flat = full_v_cpu[token_indices, head_indices, :]

    return unhit_k_flat, unhit_v_flat, id_lists


def reconstruct_unhit_only_on_gpu(
    unhit_id_map_gpu: torch.Tensor,
    global_unhit_kv_gpu: torch.Tensor,
    head_dim: int,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    assert unhit_id_map_gpu.device == device, "unhit_id_map must be on GPU!"
    assert global_unhit_kv_gpu.device == device, "global_unhit_kv must be on GPU!"

    L, BH = unhit_id_map_gpu.shape
    if L == 0:
        return torch.zeros(0, BH, head_dim, device=device, dtype=global_unhit_kv_gpu.dtype)

#     print(f"reconstruct_unhit_only_on_gpu unhit_id_map_gpu shape = {unhit_id_map_gpu.shape}")
#     print(f"reconstruct_unhit_only_on_gpu global_unhit_kv_gpu shape = {global_unhit_kv_gpu.shape}")
    
    output = F.embedding(unhit_id_map_gpu, global_unhit_kv_gpu)
    return output
