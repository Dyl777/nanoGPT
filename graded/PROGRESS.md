# Progress: OPTAL Graded LLM papers

Read this file before continuing. Update `graded/PROGRESS.json` in the same change.

## How OPTAL Graded fits

OPTAL keeps multiple implementations alive (`@Technique` / `@Impl`) and scores them with `@Graded` blocks. Each Graded block compiles to a `Benchmark_*` class whose `run()` records `endpoints`. This corpus turns **LLM-paper analyses** into those Graded endpoints so a path is judged by the same axes papers use (quality, ablations, compute, masking, bidirectionality, size, transfer).

Compile (from repo root, after `go build` of optalgo):

```text
.\optalgo.exe compile graded/papers/01_1706.03762_attention.optll python
```

Run figures without compiling:

```text
python -m graded.run_graded --papers 1,2
```

## Completed

| # | arXiv | Paper | Status |
|---|-------|--------|--------|
| 1 | [1706.03762](https://arxiv.org/abs/1706.03762) | Attention Is All You Need | done 2026-09-13 |
| 2 | [1810.04805](https://arxiv.org/abs/1810.04805) | BERT | done 2026-09-13 |

## Next (do not skip; increment `next_index`)

3. [1907.11692](https://arxiv.org/abs/1907.11692) RoBERTa  
4. [1910.10683](https://arxiv.org/abs/1910.10683) T5  
5. [2001.08361](https://arxiv.org/abs/2001.08361) Scaling Laws  
6. [2005.14165](https://arxiv.org/abs/2005.14165) GPT-3  

GPT-1 is not on arXiv; it is covered as the LTR baseline inside BERT Table 5.

## Continuation recipe

1. Open `PROGRESS.json`, take the queue item whose `index == next_index`.
2. Fetch the HTML/PDF on arXiv; list every figure and table.
3. Add functions in `arxiv_llm_lib.py`, a `graded/papers/NN_<id>_*.optll`, an `EXPLAIN.md`, and write figures under `graded/figures/<id>/`.
4. Recreate published numbers as plots; add a **local probe** for every mathematical claim that can run on tensors in this repo.
5. Append to `completed`, bump `next_index`, re-run `python -m graded.run_graded`.
