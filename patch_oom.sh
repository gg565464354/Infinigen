#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"

python3 - "$ROOT" <<'PY'
import os
import re
import sys
from glob import glob

root = sys.argv[1]


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def patch_flex_opt(path):
    text = read_text(path)
    original = text
    changed = False

    # Patch MLP forward with chunking if not already present.
    if "FLEXGEN_MLP_CHUNK_SIZE" not in text:
        old_block = (
            "        x = rms_norm(x, w[\"norm_weight\"], eps=self.config.rms_norm_eps)\n"
            "        gate = F.linear(x, w[\"gate_proj_weight\"])\n"
            "        up = F.linear(x, w[\"up_proj_weight\"])\n"
            "        x = gate * F.silu(up)\n"
            "        x = F.linear(x, w[\"down_proj_weight\"])\n"
            "        hidden.val = x\n"
        )
        new_block = (
            "        x = rms_norm(x, w[\"norm_weight\"], eps=self.config.rms_norm_eps)\n"
            "        # Chunk long sequences to reduce peak MLP activation memory.\n"
            "        chunk_size = int(os.environ.get(\"FLEXGEN_MLP_CHUNK_SIZE\", \"512\"))\n"
            "        if x.dim() == 3 and chunk_size > 0 and x.shape[1] > chunk_size:\n"
            "            out = x if x.is_contiguous() else torch.empty_like(x)\n"
            "            for start in range(0, x.shape[1], chunk_size):\n"
            "                end = min(start + chunk_size, x.shape[1])\n"
            "                x_chunk = x[:, start:end, :]\n"
            "                gate = F.linear(x_chunk, w[\"gate_proj_weight\"])\n"
            "                up = F.linear(x_chunk, w[\"up_proj_weight\"])\n"
            "                up = F.silu(up, inplace=True)\n"
            "                gate.mul_(up)\n"
            "                out[:, start:end, :] = F.linear(gate, w[\"down_proj_weight\"])\n"
            "            hidden.val = out\n"
            "            return\n"
            "        gate = F.linear(x, w[\"gate_proj_weight\"])\n"
            "        up = F.linear(x, w[\"up_proj_weight\"])\n"
            "        up = F.silu(up, inplace=True)\n"
            "        gate.mul_(up)\n"
            "        x = F.linear(gate, w[\"down_proj_weight\"])\n"
            "        hidden.val = x\n"
        )
        if old_block in text:
            text = text.replace(old_block, new_block, 1)
            changed = True

    # Add @torch.no_grad() before generate.
    if "@torch.no_grad()" not in text:
        if "\n    def generate(" in text:
            text = text.replace("\n    def generate(", "\n    @torch.no_grad()\n    def generate(", 1)
            changed = True

    # Free activations in generation_loop_normal.
    if "self.hidden[i][j - 1][k].val = None" not in text:
        marker = "                    self.store_cache(i, j, k, overlap=False)\n"
        if marker in text:
            insert = (
                marker
                + "                    if j > 0:\n"
                + "                        # Free previous layer activations to reduce peak memory.\n"
                + "                        self.hidden[i][j - 1][k].val = None\n"
                + "                    if j == self.num_layers - 1:\n"
                + "                        self.hidden[i][j][k].val = None\n"
            )
            text = text.replace(marker, insert, 1)
            changed = True

    if changed and text != original:
        write_text(path, text)
        print(f"Patched: {path}")
    else:
        print(f"No change: {path}")


def patch_mha_gen_qwen_infin_v2(path):
    text = read_text(path)
    original = text
    changed = False

    m = re.search(r"^    def mha_gen_qwen_infin_v2\b", text, re.M)
    if not m:
        print(f"Skip (no mha_gen_qwen_infin_v2): {path}")
        return

    start = m.start()
    m2 = re.search(r"^    def ", text[m.end():], re.M)
    end = m.end() + (m2.start() if m2 else len(text[m.end():]))
    func = text[start:end]

    if "hist_len = min(cache_len, src_s)" in func:
        print(f"No change: {path}")
        return

    # Patch the non-sparse path inside this function.
    cat_re = re.search(
        r"(?P<indent>\s*)k_all = torch.cat\(\[k_all, k_cache_new\], dim=0\)\n"
        r"(?P=indent)v_all = torch.cat\(\[v_all, v_cache_new\], dim=0\)",
        func,
    )
    if not cat_re:
        print(f"No change: {path}")
        return

    indent = cat_re.group("indent")
    replace = (
        f"{indent}# If we have a full cache buffer, update in-place and only use history up to src_s.\n"
        f"{indent}# Otherwise (prefetch/sparse cache), append the current token to the local window.\n"
        f"{indent}hist_len = min(cache_len, src_s)\n"
        f"{indent}if hist_len >= src_s and hist_len > 0:\n"
        f"{indent}    k_all = k_all[:hist_len]\n"
        f"{indent}    v_all = v_all[:hist_len]\n"
        f"{indent}    k_all[src_s - 1 : src_s] = k_cache_new\n"
        f"{indent}    v_all[src_s - 1 : src_s] = v_cache_new\n"
        f"{indent}    total_len = src_s\n"
        f"{indent}else:\n"
        f"{indent}    if hist_len > 0:\n"
        f"{indent}        k_all = k_all[:hist_len]\n"
        f"{indent}        v_all = v_all[:hist_len]\n"
        f"{indent}    k_all = torch.cat([k_all, k_cache_new], dim=0)\n"
        f"{indent}    v_all = torch.cat([v_all, v_cache_new], dim=0)\n"
        f"{indent}    total_len = k_all.shape[0]"
    )

    func = func[:cat_re.start()] + replace + func[cat_re.end():]
    func = re.sub(r"cache_len\s*\+\s*1", "total_len", func)

    text = text[:start] + func + text[end:]
    if text != original:
        write_text(path, text)
        changed = True

    if changed:
        print(f"Patched: {path}")
    else:
        print(f"No change: {path}")


def find_files(pattern):
    return sorted(glob(os.path.join(root, pattern), recursive=True))


def patch_cache_selection(path):
    text = read_text(path)
    original = text
    changed = False

    if "def gpu_cache_load_asyn_v3" in text and "prefetch_idx is None" not in text:
        needle = "        # Step 1: 获取未命中KV\n"
        if needle in text:
            insert = (
                needle
                + "        if prefetch_idx is None or prefetch_idx.numel() == 0:\n"
                + "            for i in range(len(group_cached_gpu_k)):\n"
                + "                cached_k = group_cached_gpu_k[i]\n"
                + "                cached_v = group_cached_gpu_v[i]\n"
                + "                empty_shape = (0, cached_k.shape[1], cached_k.shape[2])\n"
                + "                empty_k = torch.empty(empty_shape, device=cached_k.device, dtype=cached_k.dtype)\n"
                + "                empty_v = torch.empty(empty_shape, device=cached_v.device, dtype=cached_v.dtype)\n"
                + "                group_final_k.append((cached_k, empty_k))\n"
                + "                group_final_v.append((cached_v, empty_v))\n"
                + "            return group_final_k, group_final_v, None\n\n"
            )
            text = text.replace(needle, insert, 1)
            changed = True

    if "pad_idx_list = prefetch_idx[0][0].tolist()" in text and "pad_idx is None" not in text:
        text = text.replace(
            "        pad_idx_list = prefetch_idx[0][0].tolist()\n",
            "        if pad_idx is None or pad_idx.numel() == 0:\n"
            "            pad_idx_list = prefetch_idx[0][0].tolist()\n"
            "        else:\n"
            "            pad_idx_list = pad_idx[0][0].tolist()\n",
            1,
        )
        changed = True

    if changed and text != original:
        write_text(path, text)
        print(f"Patched: {path}")
    else:
        print(f"No change: {path}")


flex_opts = ["/root/InfiniGen/speedup/flexgen/flexgen/flex_opt.py"]
pytorch_backends = ["/root/InfiniGen/speedup/flexgen/flexgen/pytorch_backend.py"]
cache_controllers = ["/root/InfiniGen/speedup/flexgen/flexgen/cache_selection_controller_v2.py"]

for path in flex_opts:
    if os.path.exists(path):
        patch_flex_opt(path)
    else:
        print(f"Skip (missing): {path}")

for path in pytorch_backends:
    if os.path.exists(path):
        patch_mha_gen_qwen_infin_v2(path)
    else:
        print(f"Skip (missing): {path}")

for path in cache_controllers:
    if os.path.exists(path):
        patch_cache_selection(path)
    else:
        print(f"Skip (missing): {path}")
PY

chmod +x /root/InfiniGen/patch_oom.sh

echo "Done. Run: /root/InfiniGen/patch_oom.sh [repo_root]"
