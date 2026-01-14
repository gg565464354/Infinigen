import torch
import torch.nn.functional as F


def partial_weight_index_generation(query, n_head, head_dim, partial_weight_ratio):
    """Generates the indices of partial weight query and partial key cache.

    On the prefill stage, generates the indices of partial weight query and
    partial key cache using the query matrix. By comparing the absolute sum of
    each column of the query matrix, gets the indices of top-k columns. These
    columns correspond to the columns that strongly affect the attention score.
    Thus, we use only those partial columns of query and key for speculation.

    Args:
        query: Query matrix (b, n, D)
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d
        partial_weight_ratio: Ratio of the top-k columns

    Returns:
        partial_weight_index: Indices of top-k columns (b, h, d')
            where d' is d * (partial_weight_ratio).
    """

    partial_weight_index = torch.zeros(n_head, int(head_dim * partial_weight_ratio)).to(
        query.device
    )
    b = query.shape[0]

    for h_idx in range(n_head):
        start = h_idx * head_dim
        end = (h_idx + 1) * head_dim
        _, ind = torch.topk(
            torch.sum(torch.abs(query[0, :, start:end]), dim=-2),
            int(head_dim * partial_weight_ratio),
        )
        partial_weight_index[h_idx] = ind

    return partial_weight_index.unsqueeze(0).repeat(b, 1, 1).to(torch.int64)


def set_partial_cache(k_cache, partial_index, n_head, head_dim):
    """Sets the partial key cache.

    On the prefill and decoding stages, generates the partial key cache
    following the partial_index which indicates the indices of the important
    columns.

    Args:
        k_cahce: Key cache (n, bh, d)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_cache: Partial key cache (n, bh, d')
    """

    n, bh, _ = k_cache.shape
    partial_cache = torch.gather(
        k_cache.view(n, -1, n_head, head_dim),
        3,
        partial_index.unsqueeze(0).repeat(n, 1, 1, 1),
    )
    return partial_cache.view(n, bh, -1)


def set_partial_cache_gqa(raw_k_cache, partial_index, n_head, n_kv_head, head_dim):
    """
    Builds partial key cache for GQA by:
    1. Repeating the key cache from n_kv_head to n_head (GQA expansion)
    2. Gathering important columns within each head as specified by partial_index.

    Args:
        raw_k_cache: Raw key cache before repeat, shape (s, b * n_kv_head, head_dim)
        partial_index: Indices of important columns, shape (b, n_head, d_prime)
                       Each value in partial_index[b, h, :] is in [0, head_dim)
        n_head: Total number of query heads
        n_kv_head: Number of key/value heads
        head_dim: Dimension per head (d)

    Returns:
        partial_k_cache: Partial key cache after repeat and column selection, shape (s, b * n_head, d_prime)
    """
    s, total_kv_size, d = raw_k_cache.shape
    b = total_kv_size // n_kv_head  # batch size
    d_prime = partial_index.shape[-1]
    rep = n_head // n_kv_head

    assert d == head_dim, f"head_dim mismatch: got {d}, expected {head_dim}"
    assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
    assert partial_index.shape == (b, n_head, d_prime), f"partial_index shape should be (b={b}, h={n_head}, d'={d_prime})"

    # Step 1: Reshape to (s, b, n_kv_head, head_dim)
    k = raw_k_cache.view(s, b, n_kv_head, head_dim)

    # Step 2: Repeat each KV head `rep` times to match n_head
    # → (s, b, n_kv_head, rep, head_dim) → (s, b, n_head, head_dim)
    k_expanded = k.unsqueeze(3) \
                   .expand(s, b, n_kv_head, rep, head_dim) \
                   .reshape(s, b, n_head, head_dim)  # now each Q head has its K

    # Step 3: Gather important columns within each head
    # k_expanded: (s, b, n_head, head_dim)
    # partial_index: (b, n_head, d_prime)
    # We want: for each (b, h), gather k_expanded[*, b, h, :] at indices partial_index[b, h, :]

    # Expand partial_index to (s, b, n_head, d_prime)
    idx_expanded = partial_index.unsqueeze(0).expand(s, b, n_head, d_prime)  # (s, b, h, d')

    # Use torch.gather on the last dimension
    k_partial = torch.gather(
        k_expanded,      # (s, b, n_head, head_dim)
        dim=3,
        index=idx_expanded  # (s, b, n_head, d_prime)
    )  # → (s, b, n_head, d_prime)

    # Step 4: Reshape to (s, b * n_head, d_prime)
    partial_k_cache = k_partial.reshape(s, b * n_head, d_prime)

    return partial_k_cache


def set_partial_weight(w_q, partial_index, n_head, head_dim):
    """Sets the partial query weight.

    On the prefill stage, generates the partial query weight following the
    partial_index which indicates the indices of the important columns.

    Args:
        w_q: Query weight (D, D)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_weight: Partial query weight (D', D)
    """

    partial_weight = F.embedding(
        partial_index[0]
        + torch.arange(n_head)[:, None].to(partial_index.device) * head_dim,
        w_q.view(-1, w_q.shape[-1]),
    )
    return partial_weight.view(-1, w_q.shape[-1])
