import json, pathlib

nb = pathlib.Path(r"C:\Users\AMBE\Desktop\Engineering\nanoGPT\tutorial_optal.ipynb")
data = json.loads(nb.read_text(encoding="utf-8"))

# --- Build the OPTAL dispatcher cell (keeps both impls alive, selectable) ---
disp_lines = [
    "# ============================================================\n",
    "# OPTAL - matmul alternatives kept alive at EVERY matmul site\n",
    "# ============================================================\n",
    "# Source of truth (kept alive, never deleted): optal/<site>.optll\n",
    "#   linear_biasfree.optll -> Head q/k/v        (nn.Linear, bias=False)\n",
    "#   linear_bias.optll     -> proj / FeedForward / out (nn.Linear, bias=True)\n",
    "#   attn_scores.optll    -> q @ k^T\n",
    "#   attn_context.optll   -> attn_weights @ v\n",
    "# Each .optll compiles to .ipynb notebooks (optal/notebooks/) holding\n",
    "# BOTH the original matmul AND the no-multiply AdderNet alternative,\n",
    "# so neither implementation is ever removed. The original matmul\n",
    "# survives in optal/notebooks/<site>_matmul.ipynb.\n",
    "#\n",
    "# Build/runtime selection: flip _OPTAL_MATMUL_MODE to get the\n",
    "# no-matmul output, without touching the matmul path.\n",
    "import torch\n",
    "\n",
    '_OPTAL_MATMUL_MODE = "matmul"  # "matmul" = our output | "addernetmatmul" = no-multiply\n',
    "\n",
    "def optal_linear(x, w, b=None):\n",
    '    """Linear layer alternative. Keeps matmul + AdderNet L1-distance alive."""\n',
    "    bias = 0 if b is None else b\n",
    '    if _OPTAL_MATMUL_MODE == "addernetmatmul":\n',
    "        # y_o = -sum_i |x_i - w_{o,i}|  (no multiplication)\n",
    "        return -((x.unsqueeze(1) - w.unsqueeze(0)).abs().sum(dim=2)) + bias\n",
    "    return x @ w.t() + bias  # original matmul\n",
    "\n",
    "def optal_attn_scores(q, k, hs):\n",
    '    """Attention scores alternative. Keeps q@k^T + AdderNet L1-distance alive."""\n',
    "    scale = hs ** 0.5\n",
    '    if _OPTAL_MATMUL_MODE == "addernetmatmul":\n',
    "        # S_{i,j} = -sum_d |q_{i,d} - k_{j,d}|  (no multiplication)\n",
    "        return -((q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(dim=-1)) / scale\n",
    "    return (q @ k.transpose(-2, -1)) / scale  # original matmul\n",
    "\n",
    "def optal_attn_context(attn, v):\n",
    '    """Context aggregation alternative. Keeps attn@v + AdderNet L1-distance alive."""\n',
    '    if _OPTAL_MATMUL_MODE == "addernetmatmul":\n',
    "        # C_{i,d} = -sum_j |attn_{i,j} - v_{j,d}|  (no multiplication)\n",
    "        return -((attn.unsqueeze(-1) - v.unsqueeze(-2)).abs().sum(dim=-2))\n",
    "    return attn @ v  # original matmul\n",
    "\n",
    "# Mapping of EVERY matmul site in this model to its OPTAL alternative:\n",
    "#   Head.key/query/value(x) -> optal_linear(x, W, b=None)   (linear_biasfree.optll)\n",
    "#   MultiHead.proj(attn_out)  -> optal_linear(attn_out, W, b)  (linear_bias.optll)\n",
    "#   FeedForward.layers[0/2]  -> optal_linear(..., W, b)       (linear_bias.optll)\n",
    "#   NanoGPT.out(pos_enc)       -> optal_linear(pos_enc, W, b)    (linear_bias.optll)\n",
    "#   q @ k.transpose(2,1)      -> optal_attn_scores(q, k, head_sz) (attn_scores.optll)\n",
    "#   attn_weights @ v         -> optal_attn_context(attn_weights, v) (attn_context.optll)\n",
]

disp_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": disp_lines,
}

cells = data["cells"]

# Insertion point: right before the model (first cell defining self.key = nn.Linear)
insert_at = len(cells)
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "self.key = nn.Linear" in src:
        insert_at = i
        break

cells.insert(insert_at, disp_cell)

# Route the two attention @ matmuls through the OPTAL dispatcher (matmul is default -> identical behavior)
repl = [
    ("k_q_sim = q @ k.transpose(2, 1) / np.sqrt(head_sz)",
     "k_q_sim = optal_attn_scores(q, k, head_sz)  # OPTAL: matmul default, addernetmatmul kept alive"),
    ("k_q_sim = q @ k.transpose(2, 1) / np.sqrt(self.head_sz)",
     "k_q_sim = optal_attn_scores(q, k, self.head_sz)  # OPTAL: matmul default, addernetmatmul kept alive"),
    ("attn_out = attn_weights @ v",
     "attn_out = optal_attn_context(attn_weights, v)  # OPTAL: matmul default, addernetmatmul kept alive"),
]
for c in cells:
    if c.get("cell_type") != "code":
        continue
    src = c.get("source", [])
    joined = "".join(src)
    new_joined = joined
    for a, b in repl:
        if a in new_joined:
            new_joined = new_joined.replace(a, b)
    if new_joined != joined:
        # keep line structure: split back into lines
        c["source"] = [line + "\n" for line in new_joined.split("\n")]
        if c["source"] and c["source"][-1] == "\n":
            c["source"][-1] = c["source"][-1][:-1]  # drop trailing newline on last line

data["cells"] = cells
nb.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
print("inserted dispatcher at index", insert_at, "| cells now", len(cells))
