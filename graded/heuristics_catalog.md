# Heuristic catalog (LLM-paper analysis axes)

A model that is strong on every axis that LLM papers actually measure is the design goal of this Graded corpus. New papers should map onto these names rather than inventing one-off metrics when possible.

## Architecture

| Heuristic | Why papers use it | First source here |
|---|---|---|
| Scaled dot-product attention | Unscaled dots saturate softmax as d_k grows | 1706.03762 §3.2.1 |
| Multi-head split/concat | Subspace diversity without extra FLOPs | 1706.03762 §3.2.2 |
| Causal vs bidirectional mask | Autoregression vs fused left+right context | 1706.03762 decoder; 1810.04805 BERT vs GPT |
| Residual + LayerNorm | Trainable depth | 1706.03762 encoder/decoder |
| Position-wise FFN (d_ff=4 d_model) | Capacity after mixing | 1706.03762 §3.3; BERT note |
| Sinusoidal vs learned PE | Order without recurrence; extrapolation | 1706.03762 §3.5 Table 3E |
| Token+segment+position sum | Packed sentence pairs | 1810.04805 Fig. 2 |
| Weight tying embed/softmax | Parameter efficiency | 1706.03762 §3.4 |

## Training recipe

| Heuristic | Why | First source |
|---|---|---|
| Adam β2=0.98, ε=1e-9 + inverse-sqrt LR + warmup | Stable Transformer training | 1706.03762 Eq. 3 |
| Residual dropout | Overfitting | 1706.03762 Table 3D |
| Label smoothing | BLEU vs PPL trade | 1706.03762 §5.4 |
| MLM 15% with 80/10/10 | Deep bidirectionality without self-leak | 1810.04805 §3.1 |
| NSP 50/50 | Sentence-pair tasks | 1810.04805 §3.1 |
| Document-level corpus | Long contiguous context | 1810.04805 pre-training data |
| Fine-tune all params, tiny task head | Transfer | 1810.04805 §3.2 |

## Evaluation tasks (quality)

BLEU (WMT), per-wordpiece PPL, constituency F1, GLUE suite (MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE), SQuAD EM/F1, SQuAD 2.0 null answers, SWAG, NER F1.

## Compute and structure

Table 1 path length / sequential ops / layer complexity. Training FLOPs vs quality. Batch token budget. Checkpoint averaging, beam size, length penalty.

## Ablations papers always run

Heads, d_k, depth N, d_model, d_ff, dropout, PE type, pretrain objective (MLM vs LTR, NSP on/off), model size vs small-task accuracy, feature-based vs full fine-tune.

## Interpretability

Attention-head specialization, long-distance dependencies, anaphora (1706 appendix figures — qualitative; local probe is causal leak + PE relative shift).

## Later papers (queue) will add

Dynamic masking, span corruption, scaling laws (loss vs N, D, C), few-shot vs fine-tune, compute-optimal tokens/params, RLHF preference win-rate, contamination, long-context, mixture-of-experts, tokenizer fertility.
