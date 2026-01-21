import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor

import sys
import math

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
                 gpu_cache_pred=2,
                 cpu_cache_pred=2
                ):
        """
        初始化 CacheManager，使用字典保存不同 layer_id 对应的缓存数据。
        remove update pred
        """
        self._caches = {}  # key: layer_id, value: cache data
        self._update_recode = {}

        # refresh interval for cache pool update
        self._update_pred = 6
        
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
        self.gpu_cache_pred = float(gpu_cache_pred)
        self.cpu_cache_pred = float(cpu_cache_pred)
        self._max_cpu_cached_len = max(1, int(math.ceil(max_sparse_len * self.cpu_cache_pred)))
        self._max_gpu_cached_len = max(1, int(math.ceil(max_sparse_len * self.gpu_cache_pred)))
        # self._max_cached_len = 0
        # self._pinned_cache_shape = {}

        # gpu cache
        self._use_gpu_cache = {} # should be initial in create
        self._gpu_cached_group_kv = {}
        # cpu cache pool
        self._cpu_cached_group_kv = {}

        # data ana
        self.cnt = 0

        # update according to max cache len
        self._require_update = {}
        self.max_unhit_rate = 0.4 # max 40% unhit kv

        # cache map bookkeeping
        self.max_token_len = 0
        self.layer_cache_maps = {}
        self.layer_unhit_maps = {}
        self.layer_unhit_idxs = {}
        self.layer_unhit_lengths = {}
        self.layer_sparse_maps = {}
        
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
            
    

    def init_basic_group_head_ids(self, layer_group_head_ids):
        for l in range(len(layer_group_head_ids)):
            self._basic_group_head_ids[l] = layer_group_head_ids[l]    

    def add_cache(self, device, layer_id, batch_size, head_num, sparse_len, hidden_size,
                  use_gpu_cache=True, max_token_len=None):
        """
        添加或更新指定 layer_id 的缓存数据。
        
        :param layer_id: 缓存标识符（字符串）
        """
        if layer_id in self._caches:
            raise ValueError("Python ERROR! [add_cache] layer id is used!")
            return 1

        new_cache, new_class_group_id = self.create_cache_instance(device, layer_id, batch_size, head_num, sparse_len, hidden_size)

        self._caches[layer_id] = new_cache
        self._cur_group_ids[layer_id] = new_class_group_id
        self._update_recode[layer_id] = 0
        self._require_update[layer_id] = True # 第一次load必须更新
        if max_token_len is None:
            max_token_len = sparse_len
        self.max_token_len = max(self.max_token_len, int(max_token_len))

        # 在这里决定他是否要使用gpu cache
        self._use_gpu_cache[layer_id] = use_gpu_cache
        
        # if cpu_cache build  pinned kv pool space on cpu
        if use_gpu_cache:
            # gpu cache only require (max_sparse_len, bh, D) shape pinned pool
            pinned_cache_shape = (self.max_sparse_len, self.bh, self.head_dim)
            k_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
            v_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
            self._pinned_cached_k_list[layer_id] = [k_cache_space]
            self._pinned_cached_v_list[layer_id] = [v_cache_space]
        else:
            # cpu cache require (max_sparse_len, bh, D) shape pinned pool
            cpu_cache_pred = self._max_cpu_cached_len
            pinned_cache_shape = (cpu_cache_pred, self.bh, self.head_dim)
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
        return len(self._caches)

    def get_layer_group_id(self, layer_id):
        return self._cur_group_ids[layer_id]
    
    def get_layer_cache(self, layer_id):
        return self._caches[layer_id]

    def cache_miss_check(self, layer_id, prefetch_idx):
        """
        Return prefetch indices that are not in cache map; also record unhit maps.
        """
        if self._require_update.get(layer_id, False):
            return prefetch_idx

        if self.max_token_len <= 0:
            return prefetch_idx

        if layer_id not in self.layer_cache_maps or self.layer_cache_maps[layer_id] is None:
            self.layer_cache_maps[layer_id] = torch.zeros(
                (self.bh, self.max_token_len),
                device="cuda",
                dtype=torch.bool,
            )
        W_cache = self.layer_cache_maps[layer_id]

        W_sparse_gpu = build_sparse_map(prefetch_idx, self.max_token_len, device="cuda")
        W_unhit_gpu = cache_miss_detection(W_cache, W_sparse_gpu)
        unhit_idx_gpu, lengths = unhit_map_to_padded_idx(W_unhit_gpu)

        self.layer_unhit_lengths[layer_id] = lengths.cpu()
        self.layer_unhit_maps[layer_id] = W_unhit_gpu.cpu()
        self.layer_unhit_idxs[layer_id] = unhit_idx_gpu.cpu()
        self.layer_sparse_maps[layer_id] = W_sparse_gpu

        # Only update cache map for GPU cache path.
        if self._use_gpu_cache.get(layer_id, False):
            update_cache_map_(W_cache, W_sparse_gpu)

        return unhit_idx_gpu.cpu()


    def cpu_cache_load_asyn_v2(self, layer_id, transfer_stream, unhit_idx, pad_idx, all_k, all_v, dtype):
        if layer_id not in self._cpu_cached_group_kv:
            group_gpu_k, group_gpu_v, _, _ = self.load_and_update_cpu_cache(
                layer_id, transfer_stream, unhit_idx, all_k, all_v, dtype
            )
            return group_gpu_k, group_gpu_v, None

        cur_unhit_idx = unhit_idx.cpu()
        if pad_idx is None:
            pad_idx = torch.zeros((1, 1, unhit_idx.shape[2]), dtype=torch.int32)
        cur_pad_idx = pad_idx.cpu()
        cur_unhit_length = self.layer_unhit_lengths.get(layer_id)

        group_cached_k, group_cached_v = self._cpu_cached_group_kv[layer_id]
        max_unhit_len = unhit_idx.shape[0]
        cur_pad_idx_list = cur_pad_idx[0][0].int().tolist()
        if cur_unhit_length is None:
            cur_unhit_length_list = [max_unhit_len] * self.bh
        else:
            cur_unhit_length_list = cur_unhit_length.int().tolist()

        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.cuda(non_blocking=True) for v in group_cached_v]

            if max_unhit_len > 0:
                _C.select_kv_tensor_with_pad(
                    cur_unhit_idx,
                    cur_unhit_length_list,
                    cur_pad_idx_list,
                    all_k,
                    all_v,
                    self._pinned_unhit_k_list[0],
                    self._pinned_unhit_v_list[0],
                )

                tmp_unhit_k = self._pinned_unhit_k_list[0][:max_unhit_len, :, :]
                tmp_unhit_v = self._pinned_unhit_v_list[0][:max_unhit_len, :, :]
                unhit_gpu_k = [tmp_unhit_k.cuda(non_blocking=True)]
                unhit_gpu_v = [tmp_unhit_v.cuda(non_blocking=True)]
                group_final_k = [(group_cached_gpu_k[0], unhit_gpu_k[0])]
                group_final_v = [(group_cached_gpu_v[0], unhit_gpu_v[0])]
            else:
                cached_k = group_cached_gpu_k[0]
                empty_shape = (0, cached_k.shape[1], cached_k.shape[2])
                empty_k = torch.empty(empty_shape, device=cached_k.device, dtype=cached_k.dtype)
                empty_v = torch.empty(empty_shape, device=cached_k.device, dtype=cached_k.dtype)
                group_final_k = [(group_cached_gpu_k[0], empty_k)]
                group_final_v = [(group_cached_gpu_v[0], empty_v)]

        return group_final_k, group_final_v, None


    def gpu_cache_load_asyn_v3(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]

        group_cached_gpu_k, group_cached_gpu_v = self._gpu_cached_group_kv[layer_id]

        # result dtype = [(gpu_cached_k, gpu_cached_v)] (group_num)
        group_final_k = []
        group_final_v = []

        # Step 1: 获取未命中KV
        if prefetch_idx is None or prefetch_idx.numel() == 0 or prefetch_idx.shape[0] == 0:
            # No tokens to prefetch; return cached KV with empty unhit tensors.
            for i in range(len(group_cached_gpu_k)):
                cached_k = group_cached_gpu_k[i]
                empty_shape = (0, cached_k.shape[1], cached_k.shape[2])
                empty_k = torch.empty(empty_shape, device=cached_k.device, dtype=cached_k.dtype)
                empty_v = torch.empty(empty_shape, device=cached_k.device, dtype=cached_k.dtype)
                group_final_k.append((cached_k, empty_k))
                group_final_v.append((group_cached_gpu_v[i], empty_v))
            return group_final_k, group_final_v, []

        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        if pad_idx is None:
            pad_idx = torch.zeros((1, 1, prefetch_idx.shape[2]), dtype=torch.int32)
        pad_idx_list = pad_idx[0][0].int().tolist()
        group_unhit = cur_cache.get_unhit_kv_tensor_v7(
            prefetch_idx_int, 
            pad_idx_list,
            all_k, 
            all_v, 
            self._pinned_unhit_k_list, 
            self._pinned_unhit_v_list
        )

        # update cache map with unhit
        cur_cache.direct_update_cache_map_with_group_unhit(group_unhit)


        # Step 2: 获取传输完成的结果
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            if isinstance(group_unhit[i][0], list):
                tmp_unhit_len = max([len(unhit) for unhit in group_unhit[i]])
            else:
                tmp_unhit_len = max(group_unhit[i])

            tmp_unhit_k = self._pinned_unhit_k_list[i][:tmp_unhit_len, :, :]
            tmp_unhit_v = self._pinned_unhit_v_list[i][:tmp_unhit_len, :, :]
            group_unhit_k.append(tmp_unhit_k)
            group_unhit_v.append(tmp_unhit_v)

        

        # Step 3: unhit_k/v -> GPU 传输 + 收集结果
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_gpu_k)):
                group_final_k.append((group_cached_gpu_k[i], unhit_gpu_k[i]))
                group_final_v.append((group_cached_gpu_v[i], unhit_gpu_v[i]))

        # transfer_stream.synchronize()

        # step4: 判断是否需要更新
        cur_cache_len = group_cached_gpu_k[0].shape[0]
        max_unhit_len = max([unhit_k.shape[0] for unhit_k in group_unhit_k])
        # 如果剩余cache空间足够，更新cache
        if self._max_gpu_cached_len < (cur_cache_len + max_unhit_len):
            print(f"require update = True, _max_cached_len={self._max_gpu_cached_len} cur_cache_len={cur_cache_len} max_unhit_len={max_unhit_len}", flush = True)
            self._require_update[layer_id] = True


        return group_final_k, group_final_v, group_unhit

 
    
    def load_and_update_gpu_cached(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        # print(f"Layer #{layer_id} use gpu cache")
        # cur_group_ids = self._cur_group_ids[layer_id]
        
        
        # step 1: 如果预留的pinned tensor不够大, 创建新的pinned tensor
        if layer_id not in self._pinned_cached_k_list:
            pinned_cache_shape = (self.max_sparse_len, self.bh, self.head_dim)
            k_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
            v_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
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

        if self.max_token_len > 0:
            self.layer_cache_maps[layer_id] = build_sparse_map(prefetch_idx, self.max_token_len, device="cuda")
        
        # Step 5: 发起 cache更新任务
        cur_cache = self._caches[layer_id]
        future_unhit = self.executor.submit(cur_cache.update_group_cache, prefetch_idx, final_cpu_k, final_cpu_v)
        
        # Step 6: 并行开始 KV 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_gpu_k = [k.cuda(non_blocking=True) for k in final_cpu_k]
            group_gpu_v = [v.cuda(non_blocking=True) for v in final_cpu_v]


        # Step 7: 等待 CPU 计算结束
        try:
            stat = future_unhit.result()
            if stat != 0:
                raise ValueError("Python ERROR! [load_and_update_cache] Update cache Fail!")
        except Exception as e:
            print(f"future_unhit error: {e}")

        # 更新gpu cache
        if layer_id in self._gpu_cached_group_kv:
            del self._gpu_cached_group_kv[layer_id]
        self._gpu_cached_group_kv[layer_id] = (group_gpu_k, group_gpu_v)

        return group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v


    def load_and_update_cpu_cache(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype):
        if layer_id not in self._pinned_cached_k_list:
            pinned_cache_shape = (self.max_sparse_len, self.bh, self.head_dim)
            k_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
            v_cache_space = torch.empty(pinned_cache_shape, dtype=self.dtype, pin_memory=True)
            self._pinned_cached_k_list[layer_id] = [k_cache_space]
            self._pinned_cached_v_list[layer_id] = [v_cache_space]

        layer_pinned_k_space = self._pinned_cached_k_list[layer_id][0]
        layer_pinned_v_space = self._pinned_cached_v_list[layer_id][0]
        cur_sparse_len = prefetch_idx.shape[0]
        final_cpu_k = [layer_pinned_k_space[:cur_sparse_len]]
        final_cpu_v = [layer_pinned_v_space[:cur_sparse_len]]

        selected_k, selected_v = select_kv(prefetch_idx, all_k, all_v)
        group_cpu_k = [selected_k]
        group_cpu_v = [selected_v]

        final_cpu_k[0].copy_(selected_k)
        final_cpu_v[0].copy_(selected_v)

        if self.max_token_len > 0:
            self.layer_cache_maps[layer_id] = build_sparse_map(prefetch_idx, self.max_token_len, device="cuda")

        with torch.cuda.stream(transfer_stream):
            group_gpu_k = [final_cpu_k[0].cuda(non_blocking=True)]
            group_gpu_v = [final_cpu_v[0].cuda(non_blocking=True)]

        self._cpu_cached_group_kv[layer_id] = (final_cpu_k, final_cpu_v)

        return group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v
    

    def generate_class_group_ids(self, layer_id, batch_size):
        """
        Require static offline head group
        """
        # print("layer id = ", layer_id, type(layer_id))
        # print("self._basic_group_head_ids[layer_id] = ", self._basic_group_head_ids[layer_id])

        cur_basic_head_ids = self._basic_group_head_ids[layer_id]
        cur_head_num = self.layer_head_number
        final_head_ids = [[] for g in range(len(cur_basic_head_ids))]
        
        for b in range(batch_size): # 每个batch的head id 依次添加
            for g in range(len(cur_basic_head_ids)): # 遍历每个group
                for head_id in cur_basic_head_ids[g]: # 添加所有的head id，最大值为 bh*cur_head_num
                    final_head_ids[g].append(head_id + b*cur_head_num)

        return final_head_ids


    # 初始化c++模块中的cache
    def create_cache_instance(self, device, layer_id, batch_size, head_num, sparse_len, hidden_size=128):
        """ Initial the cache
        Call for every new batch of requests
        Building cache for every decoder layer.
        """
        if len(self._basic_group_head_ids) == 0:
            raise ValueError("Python ERROR! [create_cache_instance] basic_group_ids was not initialed!")

        # 构造 class group id
        new_class_group_id = self.generate_class_group_ids(layer_id, batch_size)
        new_class_group_id_tensor = [
            torch.tensor(head_id_list, dtype=torch.int32).to(device) for head_id_list in new_class_group_id    
        ]
        
        bh = batch_size*head_num
        cache_shape = (bh, sparse_len, hidden_size)
        cache_idx = torch.randint(low=0, high=sparse_len, size=(sparse_len, 1, batch_size*head_num)).to(torch.int32)

        # 初始化一个cache
        cache = _C.CPUCache(bh, cache_idx, cache_shape, new_class_group_id)
        
        return cache, new_class_group_id_tensor

    def update_gpu_cache_with_new_tensor(self, layer_id, new_k_cache, new_v_cache):
        '''
            Directly update cache.
            Be called before attention.
            The cache map is update when get unhit.
        '''
        if layer_id not in self._use_gpu_cache:
            return
        
        self._gpu_cached_group_kv[layer_id] = (new_k_cache, new_v_cache)



    # 统一的更新和加载接口
    def unified_load_api(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype):
        if self._use_gpu_cache.get(layer_id, False):
            # 判断是否需要更新
            if self._require_update[layer_id]:
                # 直接更新cache
                self._update_recode[layer_id] = 0
                self._require_update[layer_id] = False
                group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_gpu_cached(layer_id, transfer_stream, prefetch_idx, all_k, all_v)
                return (group_gpu_k, group_gpu_v, None)
            else:
                self._update_recode[layer_id] += 1
                
                # version 3
                group_final_k, group_final_v, group_unhit = self.gpu_cache_load_asyn_v3(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v)
                
                return (group_final_k, group_final_v, group_unhit)
        else:
            # 判断是否需要更新
            if self._require_update[layer_id]:
                # 直接更新cache
                self._update_recode[layer_id] = 1
                self._require_update[layer_id] = False
                group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_cpu_cache(layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype)
                
                return (group_gpu_k, group_gpu_v, None)
            else:
                
                self._update_recode[layer_id] += 1
                unhit_idx = self.cache_miss_check(layer_id, prefetch_idx)
                group_final_k, group_final_v, group_unhit = self.cpu_cache_load_asyn_v2(
                    layer_id, transfer_stream, unhit_idx, pad_idx, all_k, all_v, dtype
                )

                if self._update_recode[layer_id] > self._update_pred:
                    self._require_update[layer_id] = True

                return (group_final_k, group_final_v, group_unhit)


        

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


def build_sparse_map(sparse_ids, token_len, device="cuda"):
    """sparse_ids: (n', 1, BH) or (n', BH) -> (BH, token_len)"""
    if token_len <= 0:
        return torch.zeros((1, 0), dtype=torch.bool, device=device)
    if sparse_ids.dim() == 3:
        sparse_ids = sparse_ids.squeeze(1)
    cur_sparse_ids = sparse_ids.to(device)
    bh = cur_sparse_ids.size(1)
    W_sparse = torch.zeros((bh, token_len), dtype=torch.bool, device=device)
    sparse_ids_t = cur_sparse_ids.t()
    W_sparse.scatter_(1, sparse_ids_t.long(), True)
    return W_sparse


def cache_miss_detection(W_cache, W_sparse):
    return (~W_cache) & W_sparse


def update_cache_map_(W_cache, W_sparse):
    W_cache.bitwise_or_(W_sparse)
    return W_cache


def unhit_map_to_padded_idx(unhit_map):
    """
    Input: unhit_map (BH, T) on GPU.
    Output: padded_idx (max_n, 1, BH) on GPU and per-head lengths.
    """
    bh, t = unhit_map.shape
    device = unhit_map.device
    union_dtype = torch.int32

    arange = torch.arange(t, device=device, dtype=union_dtype).expand(bh, t)
    indices = torch.where(unhit_map, arange, torch.tensor(0, device=device))

    lengths = unhit_map.sum(dim=1)
    max_n = lengths.max().item()
    if max_n == 0:
        return torch.zeros((0, 1, bh), dtype=torch.long, device=device), lengths

    sort_keys = unhit_map.logical_not().to(torch.int32)
    _, sorted_indices = torch.sort(sort_keys, dim=1, stable=True)
    sorted_vals = torch.gather(indices, 1, sorted_indices)
    padded = sorted_vals[:, :max_n]
    padded_idx = padded.t().unsqueeze(1).int()
    return padded_idx, lengths


def reconstruct_unhit_only_on_gpu(unhit_id_map_gpu, global_unhit_kv_gpu, head_dim, device=None):
    if device is None:
        device = global_unhit_kv_gpu.device
    if unhit_id_map_gpu.device != device:
        raise ValueError("unhit_id_map must be on the same device as global_unhit_kv")
    if global_unhit_kv_gpu.device != device:
        raise ValueError("global_unhit_kv must be on the same device as unhit_id_map")

    L, bh = unhit_id_map_gpu.shape
    if L == 0:
        return torch.zeros((0, bh, head_dim), device=device, dtype=global_unhit_kv_gpu.dtype)
    return F.embedding(unhit_id_map_gpu, global_unhit_kv_gpu)
