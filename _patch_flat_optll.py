"""Surgical edits to nanoGPT_flat.ipynb.optll: paper-faithful fixes + in-place calls."""
from pathlib import Path

P = Path("nanoGPT_flat.ipynb.optll")
s = P.read_text(encoding="utf-8")

# --- paper-faithful: EncoderBlock must use ln1 (pre-norm, same as decoder Block) ---
old_enc = (
    '        "        out = torch.cat([h(x) for h in self.heads], dim=-1)\\n",\n'
    '        "        x = x + self.proj(out)\\n",\n'
    '        "        x = x + self.ffwd(self.ln2(x))\\n",'
)
new_enc = (
    '        "        out = torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1)\\n",\n'
    '        "        x = x + self.proj(out)\\n",\n'
    '        "        x = x + self.ffwd(self.ln2(x))\\n",'
)
n_enc = s.count(old_enc)
s = s.replace(old_enc, new_enc)
print("encoderblock_ln1", n_enc)

# --- paper-faithful: decoder_blocks must be a ModuleList (parameters owned) ---
old_dec = (
    '        "        self.decoder_blocks = [\\n",\n'
    '        "            CrossBlock(emb_dim, n_heads) for _ in range(n_layers)\\n",\n'
    '        "        ]\\n",'
)
new_dec = (
    '        "        self.decoder_blocks = nn.ModuleList([\\n",\n'
    '        "            CrossBlock(emb_dim, n_heads) for _ in range(n_layers)\\n",\n'
    '        "        ])\\n",'
)
n_dec = s.count(old_dec)
s = s.replace(old_dec, new_dec)
print("decoder_modulelist", n_dec)

# --- extractor: also hook decoder_blocks on encoder-decoder ---
old_ex = (
    '        "    blocks = list(model.blocks) if hasattr(model, \\"blocks\\") else []\\n",\n'
    '        "    if hasattr(model, \\"encoder\\"):\\n",\n'
    '        "        blocks = list(model.encoder) + blocks\\n",'
)
new_ex = (
    '        "    from nanogpt_pipeline import decoder_stack as _dstk, compat_model as _cm\\n",\n'
    '        "    _cm(model)\\n",\n'
    '        "    blocks = list(_dstk(model))\\n",\n'
    '        "    if hasattr(model, \\"encoder\\"):\\n",\n'
    '        "        blocks = list(model.encoder) + [b for b in blocks if b not in list(model.encoder)]\\n",'
)
if old_ex not in s:
    raise SystemExit("extract_last_token_states pattern not found")
s = s.replace(old_ex, new_ex, 1)
print("extractor patched")

# --- import nanogpt_pipeline next to bench_utils ---
old_imp = (
    '        "from bench_utils import (bench_report, bench_ids, bench_pairs_fn,\\n",\n'
    '        "                        bench_tasks, bench_niah)\\n",\n'
    '        "print(\'bench_utils loaded\')\\n",\n'
    '        "\\n",\n'
    '        "# ### Benchmark call-site setup: shared inputs live in bench_utils.py\\n",\n'
    '        "# BENCH_RESULTS collects one status record per runner so the notebook can\\n",\n'
    '        "# report what actually ran at the end.\\n",\n'
    '        "BENCH_RESULTS = []"'
)
new_imp = (
    '        "from bench_utils import (bench_report, bench_ids, bench_pairs_fn,\\n",\n'
    '        "                        bench_tasks, bench_niah)\\n",\n'
    '        "from nanogpt_pipeline import (\\n",\n'
    '        "    compat_model, live_call, run_attention_pipeline, run_bert_pipeline,\\n",\n'
    '        "    run_layermix_live, run_saber_live, run_convmem_live, run_grokking_live,\\n",\n'
    '        "    run_rilm_live, run_ignorance_live)\\n",\n'
    '        "compat_model(model)\\n",\n'
    '        "print(\'bench_utils + nanogpt_pipeline loaded\')\\n",\n'
    '        "\\n",\n'
    '        "# Shared inputs: bench_utils. Live runners/curves/compat: nanogpt_pipeline\\n",\n'
    '        "# (not graded/arxiv_llm_lib — that file only replots published PDF numbers).\\n",\n'
    '        "BENCH_RESULTS = []"'
)
if old_imp not in s:
    raise SystemExit("bench_utils import cell not found")
s = s.replace(old_imp, new_imp, 1)
print("imports patched")

# --- rewrite existing bench_report call sites to live_call(..., ns=globals()) ---
s = s.replace(
    "BENCH_RESULTS.append(bench_report(\\n\",\n        \"",
    "BENCH_RESULTS.append(live_call(\\n\",\n        \"",
)
s = s.replace(
    "    BENCH_RESULTS.append(bench_report(\\n\",\n        \"        '",
    "    BENCH_RESULTS.append(live_call(\\n\",\n        \"        '",
)
print("bench_report -> live_call replacements done")

# Insert ns=globals() before the closing of live_call that lack it is handled
# by live_call looking at kwargs; we still inject on the four in-section new calls.

def code_cell(src: str) -> str:
    lines = src.split("\n")
    body = []
    for i, line in enumerate(lines):
        esc = line.replace("\\", "\\\\").replace('"', '\\"')
        comma = "," if i < len(lines) - 1 else ""
        body.append(f'        "{esc}\\n"{comma}')
    # last line without extra newline if empty handled
    inner = ",\n".join(body)
    return (
        "    {\n"
        '      "cell_type": "code",\n'
        '      "execution_count": null,\n'
        '      "metadata": {},\n'
        '      "outputs": [],\n'
        f'      "source": [\n{inner}\n'
        "      ]\n"
        "    }"
    )


def md_cell(src: str) -> str:
    lines = src.split("\n")
    body = []
    for i, line in enumerate(lines):
        esc = line.replace("\\", "\\\\").replace('"', '\\"')
        comma = "," if i < len(lines) - 1 else ""
        body.append(f'        "{esc}\\n"{comma}')
    inner = ",\n".join(body)
    return (
        "    {\n"
        '      "cell_type": "markdown",\n'
        '      "metadata": {},\n'
        f'      "source": [\n{inner}\n'
        "      ]\n"
        "    }"
    )


def insert_after(hay: str, marker: str, insertion: str, label: str) -> str:
    i = hay.find(marker)
    if i < 0:
        raise SystemExit(f"marker not found: {label}")
    j = i + len(marker)
    return hay[:j] + insertion + hay[j:]


# After PosEnc::sinusoidal impl — Attention + BERT live pipelines belong here
# (paper 1 is the PE/attention substrate; paper 2 is the bidirectional encoder).
posenc_end = (
    '        "        return self.pe.to(idx.device)[idx]"\n'
    "      ]\n"
    "    }\n"
    "    }@},"
)
attn_bert_cells = (
    "\n"
    + md_cell(
        "## Paper 1 live pipeline — Attention Is All You Need (arXiv:1706.03762)\n"
        "\n"
        "Eq.(1) is already `Attn::vanillaatten`. This cell **calls** `nanogpt_pipeline.run_attention_pipeline` "
        "(not `graded/arxiv_llm_lib`): Eq.(3) LR curve, Table-1 cost on live dims, Table-3 head/depth PPL, "
        "Sec.3.5 sinusoidal vs learned PE, label smoothing, generation BLEU on Shakespeare. "
        "WMT BLEU stays a labeled claim."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'attention', run_attention_pipeline, ns=globals(),\n"
        "    make_model=lambda: model, ids=bench_ids(globals()),\n"
        "    int_to_token=globals().get('int_to_token')))"
    )
    + ",\n"
    + md_cell(
        "## Paper 2 live pipeline — BERT (arXiv:1810.04805)\n"
        "\n"
        "GLUE/SQuAD numbers stay in `graded/`. Here we measure the identity BERT added on this "
        "architecture: bidirectional encoder self-attn sees future tokens; the causal decoder does not. "
        "On `Arch::encoderdecoder` the encoder stack is the BERT-like path; on decoder-only the comparison "
        "is the causal mask vs an unmasked softmax of the same scores."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'bert', run_bert_pipeline, ns=globals(),\n"
        "    model=model, ids=bench_ids(globals())))"
    )
    + ","
)
s = insert_after(s, posenc_end, attn_bert_cells, "posenc_end")
print("inserted attention+bert after PosEnc")

# LayerMix call after alllayeravg_predict cell
lm_end = (
    '        "    return sum(probs) / len(probs), sorted(layer_states), {}\\n",\n'
    "      ]\n"
    "    },"
)
lm_call = (
    "\n"
    + md_cell(
        "### LayerMix live benchmark + AUROC curves\n"
        "\n"
        "Runs every probe head and aggregation rule on the frozen last-token states of **this** model "
        "(Tables 3/15/16 comparison, nanoGPT-scale). Definitions live in `nanogpt_pipeline.py`."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'layermix', run_layermix_live, ns=globals(),\n"
        "    model=model, pairs_fn=lambda: bench_pairs_fn(globals())))"
    )
    + ","
)
s = insert_after(s, lm_end, lm_call, "layermix alllayeravg")
print("inserted layermix call")

# SABER call after compatibility SABER markdown header — actually after last saber code
# Insert before "## 14. Grokking Lab"
saber_mark = '        "## 14. Grokking Lab'
saber_call = (
    md_cell(
        "### SABER live benchmark + exit curves\n"
        "\n"
        "Runs `saber_generate_early_exit` on Shakespeare stand-in tasks. Task dicts from `bench_tasks` "
        "are adapted (`prompt`/`gold` → paper `q`/`a`) in `nanogpt_pipeline` so the runner is compatible."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'saber', run_saber_live, ns=globals(),\n"
        "    model=model, encode_fn=encode, decode_fn=decode,\n"
        "    tasks=bench_tasks(globals())))"
    )
    + ",\n    {\n      \"cell_type\": \"markdown\",\n      \"metadata\": {},\n      \"source\": [\n        "
)
# The markdown cell for section 14 already starts with { cell_type markdown. We need to insert BEFORE that cell.
idx = s.find('        "## 14. Grokking Lab')
if idx < 0:
    raise SystemExit("section 14 marker missing")
# walk back to the start of this markdown cell
cell_start = s.rfind("    {\n      \"cell_type\": \"markdown\"", 0, idx)
insert = (
    md_cell(
        "### SABER live benchmark + exit curves\n"
        "\n"
        "Runs `saber_generate_early_exit` on Shakespeare stand-in tasks. Task dicts from `bench_tasks` "
        "are adapted (`prompt`/`gold` → paper `q`/`a`) in `nanogpt_pipeline` so the runner is compatible."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'saber', run_saber_live, ns=globals(),\n"
        "    model=model, encode_fn=encode, decode_fn=decode,\n"
        "    tasks=bench_tasks(globals())))"
    )
    + ",\n"
)
s = s[:cell_start] + insert + s[cell_start:]
print("inserted saber call")

# ConvMem call before "## 16. Wired Grokking"
idx = s.find('        "## 16. Wired Grokking')
if idx < 0:
    raise SystemExit("section 16 marker missing")
cell_start = s.rfind("    {\n      \"cell_type\": \"markdown\"", 0, idx)
insert = (
    md_cell(
        "### ConvMem live NIAH + F1/Sub-EM curves\n"
        "\n"
        "`bench_niah` yields dict queries; the paper runner wants `(q, gold)` pairs. "
        "`nanogpt_pipeline.as_niah` makes those inputs compatible without changing the hierarchy."
    )
    + ",\n"
    + code_cell(
        "_nd, _dd, _qq = bench_niah(globals())\n"
        "BENCH_RESULTS.append(live_call(\n"
        "    'convmem', run_convmem_live, ns=globals(),\n"
        "    needles=_nd, distractors=_dd, queries=_qq, W=40, S=16))"
    )
    + ",\n"
)
s = s[:cell_start] + insert + s[cell_start:]
print("inserted convmem call")

# Grokking call before "## 17. Compatibility Ledger"
idx = s.find('        "## 17. Compatibility Ledger')
if idx < 0:
    raise SystemExit("section 17 marker missing")
cell_start = s.rfind("    {\n      \"cell_type\": \"markdown\"", 0, idx)
insert = (
    md_cell(
        "### Grokking live D-sweep + Tgrok curve\n"
        "\n"
        "Calls `wired_d_sweep` on a **fresh** clone of the compiled architecture (not the trained "
        "Shakespeare weights). Predicted D-doubling speedup is the paper's $D^{-2.04}$."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'grokking', run_grokking_live, ns=globals(),\n"
        "    make_model_fn=lambda: model, steps=200))"
    )
    + ",\n"
)
s = s[:cell_start] + insert + s[cell_start:]
print("inserted grokking call")

# RiLM call after run_rilm_small_comparison cell (before compatibility markdown)
rilm_mark = '        "### Compatibility: RiLM:'
idx = s.find(rilm_mark)
if idx < 0:
    raise SystemExit("rilm compat marker missing")
cell_start = s.rfind("    {\n      \"cell_type\": \"markdown\"", 0, idx)
insert = (
    md_cell(
        "### RiLM live geodesic vs linear small-scale + PPL curve\n"
        "\n"
        "Paper Table I: delete Wout, tie the geodesic head to the embedding table. "
        "Incompatible with a separate untied copy (NEVER). `GeodesicHead` has no `out_features`; "
        "the small-scale runner already uses `vocab_sz` as the compatible stand-in."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'rilm', run_rilm_live, ns=globals(),\n"
        "    make_model_fn=lambda: model, get_batch_fn=get_batch, steps=50))"
    )
    + ",\n"
)
s = s[:cell_start] + insert + s[cell_start:]
print("inserted rilm call")

# Ignorance call: after ignorance_intervene cell, before compatibility markdown for ignorance
ign_mark = '        "### Compatibility: TASCO'  # wrong
# find Direction of Ignorance compatibility
idx = s.find('The Geometry of Ignorance")')
# better:
idx = s.find("### Compatibility:")
# search unique
ign_md = None
for needle in (
    "### Compatibility: The Geometry of Ignorance",
    "### Compatibility: Direction of Ignorance",
    "ignorance-direction",
):
    if needle in s:
        print("found", needle)

# After ignorance_intervene function — look for backprop/compat following section 22 code
idx = s.find("def ignorance_intervene")
if idx < 0:
    raise SystemExit("ignorance_intervene missing")
# next markdown after this function
idx2 = s.find('      "cell_type": "markdown"', idx)
cell_start = s.rfind("    {\n      \"cell_type\": \"markdown\"", idx, idx2 + 10)
# actually idx2 is inside the next cell; find the `{` of that cell
cell_start = s.rfind("\n    {\n", idx, idx2)
insert = (
    ",\n"
    + md_cell(
        "### Ignorance live OLS fit + λ curves\n"
        "\n"
        "Eq.4/9 centered OLS on the live unembedding (or shared embedding under `GeodesicHead`, "
        "the paper-compatible stand-in when Wout is gone). Corpus = Shakespeare (documented "
        "deviation from Pile; coverage c=1.0 because the char vocab is fully observed)."
    )
    + ",\n"
    + code_cell(
        "BENCH_RESULTS.append(live_call(\n"
        "    'ignorance', run_ignorance_live, ns=globals(),\n"
        "    model=model, text=text, encode=encode))"
    )
)
# insert before the markdown cell that follows the intervene code cell
# cell_start currently points at newline before `{` of markdown
s = s[: cell_start + 1] + insert + s[cell_start + 1 :]
print("inserted ignorance call")

# --- shrink section 18 dump: replace the four definition+call blocks with a pointer ---
# Keep the section header, drop duplicate runners now called in-section.
start18 = s.find('        "## 18. Benchmark Runners')
end18 = s.find("### Length note: why RULER")
if start18 < 0 or end18 < 0:
    raise SystemExit(f"section 18 bounds missing {start18} {end18}")
# include the markdown cell that contains Length note — keep from Length note onward
# Walk back from Length note to its cell `{`
len_cell = s.rfind("    {\n      \"cell_type\": \"markdown\"", start18, end18 + 10)
# Walk back from "## 18" to its cell
head18 = s.rfind("    {\n      \"cell_type\": \"markdown\"", 0, start18)
new18 = (
    md_cell(
        "## 18. Benchmark Runners (moved in-section)\n"
        "\n"
        "LayerMix / SABER / grokking / ConvMem live runners are called **next to those papers** "
        "via `nanogpt_pipeline` (not dumped here, not in `graded/arxiv_llm_lib`). "
        "Sliding-window attention remains below as the length-gap stand-in for RULER-scale NIAH."
    )
    + ",\n"
)
s = s[:head18] + new18 + s[len_cell:]
print("rewrote section 18 dump")

# --- remove section 33 Attention dump at the bottom (defs belong in nanogpt_pipeline) ---
start33 = s.find('        "## 33. Attention Is All You Need')
if start33 < 0:
    raise SystemExit("section 33 missing")
head33 = s.rfind("    {\n      \"cell_type\": \"markdown\"", 0, start33)
end_keep = s.find('        "### Relations: call sites x every runner section"')
if end_keep < 0:
    raise SystemExit("call sites summary missing")
keep_cell = s.rfind("    {\n      \"cell_type\": \"markdown\"", 0, end_keep)
s = s[:head33] + s[keep_cell:]
print("removed section 33 bottom dump")

P.write_text(s, encoding="utf-8")
print("wrote", P, "chars", len(s))
