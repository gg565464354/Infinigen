import torch


def weight_bias_concat(weight, bias, scaling=False, head_dim=1.0):
    """Concatenates the weight matrix and bias.

    On the warmup phase, concatenates the weight matrix and bias for skewing.
    This manipulation does not hurt the correctness.

    Args:
        weight: Weight matrix (D, D)
        bias: Bias vector (D)
        scaling: If ture, scales the concatenated weight and bias to skip
            the scaling after projection.
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        concatenated weight and bias (D, D+1)
    """

    if scaling:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1) * (
            head_dim**-0.5
        )
    else:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1)


def reform_hidden_states(hidden_states):
    """Concatenates the weight matrix and bias.

    Concatenates the hidden states with a column of 1.
    This reformation with the concatenated weight and bias  makes the linear
    projection into a one matrix multiplication without bias addition.

    Args:
        hidden: Hidden states (b, n, D)

    Returns:
        reformed hidden states (b, n, D+1)
    """

    return torch.cat(
        (hidden_states, torch.ones_like(hidden_states)[:, :, 1].unsqueeze(2)), dim=-1
    )


def skew(query, key, wq, wk, n_head, head_dim):
    """Manipulates the query/key weight matrix for skewing the qeury and key matrix.

    On the warmup phase, manipulates the query/key weight matrix for
    skewing the query and key matrix. By doing so, a few columns of
    the query and key matrix have become much more important. We use
    the columns for attention speculation.

    Args:
        query: Query matrix (b, n, h, d)
        key: Key matrix (b, n, h, d)
        w_q: Concatenated query weight and bias (D, D+1)
        w_k: Concatenated key weight and bias (D, D+1)
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        w_q: Manipulated w_q (D, D+1)
        w_k: Manipulated w_k (D, D+1)

    """

    for h_idx in range(n_head):
        start = h_idx * head_dim
        end = (h_idx + 1) * head_dim
        _, sq, vq = torch.svd(query[0, :, h_idx].to(torch.float))
        _, sk, _ = torch.svd(key[0, :, h_idx].to(torch.float))
        sq = sq.to(torch.float16)
        vq = vq.to(torch.float16)
        sk = sk.to(torch.float16)
        sq = sq * sk
        A = torch.zeros(head_dim, head_dim).to(query.device).to(torch.float16)
        _, ind = sq.sort()
        A = A.scatter(-1, ind.unsqueeze(0).repeat(head_dim, 1), vq)
        wq[start:end, :] = A.t() @ wq[start:end]
        wk[start:end, :] = A.t() @ wk[start:end]
    return wq, wk


def skew_gqa(q, k, w_q, w_k, n_head, n_kv_head, head_dim):
    import torch

    b, _, s, d = q.shape
    h = w_k.shape[1]  # hidden_size
    total_k_dim = n_kv_head * head_dim
    total_q_dim = n_head * head_dim
    dtype = w_q.dtype
    device = w_q.device
    assert dtype == torch.bfloat16, f"Expected bfloat16, got {dtype}"
    assert w_k.shape == (total_k_dim, h), f"Expected w_k.shape=(n_kv*d, h), got {w_k.shape}"
    assert w_q.shape == (total_q_dim, h), f"Expected w_q.shape=(n_head*d, h), got {w_q.shape}"
    assert d == head_dim
    assert n_head % n_kv_head == 0
    rep = n_head // n_kv_head

    # -------------------------------
    # Step 1: Expand w_k to full shape: (n_kv_head*d, h) -> (n_head*d, h)
    # -------------------------------
    w_k_reshaped = w_k.reshape(n_kv_head, head_dim, h)  # (n_kv, d, h)
    w_k_repeated = w_k_reshaped.unsqueeze(1) \
                                   .expand(n_kv_head, rep, head_dim, h) \
                                   .reshape(n_head, head_dim, h)  # (n_head, d, h)
    w_k_full = w_k_repeated.reshape(n_head * head_dim, h)  # (n_head*d, h)

    # -------------------------------
    # Step 2: Prepare q and k for SVD (use first sample)
    # -------------------------------
    q_flat = q[0].to(torch.float32)  # (n_head, s, d)
    k_flat = k[0].to(torch.float32)  # (n_head, s, d)

    # -------------------------------
    # Step 3: Reshape weights to (n_head, d, h) for per-head transform
    # -------------------------------
    w_q_reshaped = w_q.reshape(n_head, head_dim, h).to(torch.float32).clone()  # (n_head, d, h)
    w_k_full_reshaped = w_k_full.to(torch.float32).reshape(n_head, head_dim, h).clone()  # (n_head, d, h)

    # -------------------------------
    # Step 4: Per-head skew
    # -------------------------------
    with torch.no_grad():
        for h_idx in range(n_head):
            q_h = q_flat[h_idx]  # (s, d)
            k_h = k_flat[h_idx]  # (s, d)
            if q_h.size(0) < head_dim:
                continue

            try:
                _, s_q, V_q = torch.svd(q_h, some=True)  # (d,), (d,d)
                _, s_k, _ = torch.svd(k_h, some=True)
            except Exception as e:
                print(f"SVD failed for head {h_idx}: {e}")
                continue

            importance = s_q * s_k
            _, sorted_indices = importance.sort(descending=True)  # (d,)

            A = V_q.index_select(0, sorted_indices)  # (d, d)

            # Apply: new_basis = A @ old_basis
            w_q_reshaped[h_idx] = torch.matmul(A, w_q_reshaped[h_idx])  # (d,h)
            w_k_full_reshaped[h_idx] = torch.matmul(A, w_k_full_reshaped[h_idx])

    # -------------------------------
    # Step 5: Reshape back to original shape
    # -------------------------------
    w_q_skewed = w_q_reshaped.reshape(n_head * head_dim, h).to(dtype)  # (n_head*d, h)
    w_k_full = w_k_full_reshaped.reshape(n_head * head_dim, h)
    if rep > 1:
        w_k_grouped = w_k_full_reshaped.view(n_kv_head, rep, head_dim, h)
        w_k_skewed = w_k_grouped.mean(dim=1).reshape(n_kv_head * head_dim, h).to(dtype)
    else:
        w_k_skewed = w_k_full.to(dtype)

    return w_q_skewed, w_k_skewed
