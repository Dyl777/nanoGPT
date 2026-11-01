"""External benchmark suites (download-backed live runs on the compiled nanoGPT).

Every suite:
- downloads its dataset on first use (HF `datasets`; cached under data/external/hf),
- scores with the tokenizer that MATCHES the compiled model's vocab
  (tiktoken GPT-2 BPE for the gpt2bpe Vocab impl; the notebook's char/word
  encoder otherwise - enc_for_model picks and provenance records which),
- returns a result dict; on any failure returns {"error": "<reason>"} which the
  bench/plot layer renders as SKIP-with-reason (never faked numbers),
- embeds a provenance string (dataset/split/n/seed/shots/scoring).

Published paper magnitudes stay in graded/figures (recorded claims, never simulated).
"""
from __future__ import annotations

import json
import os
import random
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

DATA = Path("data/external")
_HF_DIR = DATA / "hf"
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_DIR))


# ---------------------------------------------------------------------------
# Encoders (must match the compiled model's vocab exactly)
# ---------------------------------------------------------------------------

class BpeEncoder:
    name = "tiktoken-gpt2"

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self.n_vocab = self._enc.n_vocab

    def encode(self, s: str) -> List[int]:
        return list(self._enc.encode(s, allowed_special=set()))

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))


class CharEncoder:
    name = "notebook-char"

    def __init__(self, encode, decode, n_vocab: int):
        self._encode = encode
        self._decode = decode
        self.n_vocab = n_vocab

    def encode(self, s: str) -> List[int]:
        return list(self._encode(s))

    def decode(self, ids: Sequence[int]) -> str:
        return self._decode(list(ids))


def enc_for_model(model, ns: dict):
    """Pick an encoder consistent with the model's embedding size, or None."""
    V = model.token_embedding.num_embeddings
    if V == 50257:
        try:
            return BpeEncoder()
        except Exception:
            return None
    itok = ns.get("int_to_token")
    if isinstance(itok, dict) and len(itok) == V and callable(ns.get("encode")):
        return CharEncoder(ns["encode"], ns["decode"], V)
    return None


def _err(msg: str) -> Dict[str, Any]:
    return {"error": str(msg)[:300]}


def _dev(model) -> torch.device:
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Core scorers
# ---------------------------------------------------------------------------

def _sum_logprob(model, ids: List[int], n_ans: int) -> float:
    """Sum log P(last n_ans ids | prefix) under the compiled model.

    Prefix-cropped to the compiled ctx: scores the continuation given the
    in-window tail of the prompt, so small-ctx paths stay measurable."""
    cap = int(model.ctx_len)
    if len(ids) > cap:
        ids = ids[-cap:]
    n_ans = min(n_ans, len(ids) - 1)
    with torch.no_grad():
        x = torch.tensor([ids], device=_dev(model))
        logits = model(x)[0]
        logp = torch.log_softmax(logits.float(), dim=-1)
        total = 0.0
        T = len(ids)
        for k in range(T - n_ans, T):
            total += float(logp[k - 1, ids[k]])
    return total


def _mc_pick(model, enc, prompt: str, candidates: Sequence[str]) -> Tuple[int, List[float]]:
    """Rank candidates by continuation log-probability (boundary-split ids)."""
    base_len = len(enc.encode(prompt))
    totals = []
    for cand in candidates:
        full = enc.encode(prompt + cand)
        n_ans = max(1, len(full) - base_len)
        totals.append(_sum_logprob(model, full, n_ans))
    return int(max(range(len(totals)), key=lambda i: totals[i])), totals


def _greedy_generate(model, enc, prompt: str, max_new: int = 256,
                     stop: Sequence[str] = ("\n\n", "\nQuestion")) -> str:
    """Greedy decode with sliding context; cuts at the first stop string."""
    ids = enc.encode(prompt)
    cut = -min(len(ids), model.ctx_len)
    ctx = ids[cut:]
    gen: List[int] = []
    for _ in range(max_new):
        with torch.no_grad():
            x = torch.tensor([ctx], device=_dev(model))
            logits = model(x)[0]
            nxt = int(torch.argmax(logits[-1]))
        gen.append(nxt)
        ctx = (ctx + [nxt])[-model.ctx_len:]
        if (len(gen) % 8 == 0) or len(gen) == max_new:
            s = enc.decode(gen)
            idx = min([s.find(p) for p in stop if s.find(p) >= 0], default=-1)
            if idx >= 0:
                gen = gen[:idx]
                break
    return enc.decode(gen)


def _prob_mass(model, enc, prompt: str, correct: Sequence[str],
               wrong: Sequence[str]) -> float:
    """Normalized probability mass on the correct answers (TruthfulQA MC2)."""
    cands = list(correct) + list(wrong)
    _i, totals = _mc_pick(model, enc, prompt, cands)
    m = max(totals)
    ex = [t - m for t in totals]
    ex = [2.718281828459045 ** t for t in ex]
    z = sum(ex) or 1.0
    return sum(ex[:len(correct)]) / z


# ---------------------------------------------------------------------------
# Datasets (download + cache, fail-soft)
# ---------------------------------------------------------------------------

def _hf(repo: str, config: Optional[str], split: str, key: str = ""):
    from datasets import load_dataset
    last = None
    for attempt in range(2):  # HF resumes generation; known transient arrow hiccups
        try:
            return load_dataset(repo, config, split=split,
                                cache_dir=str(DATA / ("hf" if not key else key)))
        except Exception as e:  # noqa: BLE001 - fail-soft contract
            last = e
    raise IOError(f"{repo}/{config}/{split} failed twice: {last}")


def _pick_repo(candidates: Sequence[Tuple[str, Optional[str]]], split: str):
    last = None
    for repo, config in candidates:
        try:
            return _hf(repo, config, split), repo
        except Exception as e:  # noqa: BLE001 - fail-soft contract
            last = e
    raise IOError(f"no dataset among {[c[0] for c in candidates]}: {last}")


# ---------------------------------------------------------------------------
# MMLU (Hendrycks et al. 2021): 5-shot likelihood MC over A/B/C/D letters
# ---------------------------------------------------------------------------

def bench_mmlu(model, enc, n: int = 250, seed: int = 0, k_shot: int = 5,
               subjects: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    try:
        enc_ok = enc is not None
        test = _hf("cais/mmlu", "all", "test")
        dev = _hf("cais/mmlu", "all", "dev") if k_shot else None
        if subjects:
            test = [r for r in test if r["subject"] in subjects]
        rng = random.Random(seed)
        rows = list(test)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        shots_by_subj: Dict[str, List[Any]] = {}
        if dev is not None:
            for r in dev:
                shots_by_subj.setdefault(r["subject"], []).append(r)

        def render(ex, shots):
            s = f"The following are multiple choice questions (with answers) about {ex['subject']}.\n\n"
            for sh in shots[:k_shot]:
                s += (f"{sh['question']}\n" + "".join(
                    f"{let}. {a}\n" for let, a in zip("ABCD", sh["choices"]))
                    + "Answer: " + "ABCD"[sh["answer"]] + "\n\n")
            s += (ex["question"] + "\n" + "".join(
                f"{let}. {a}\n" for let, a in zip("ABCD", ex["choices"]))
                + "Answer:")
            return s

        per_subj: Dict[str, List[int]] = {}
        correct = 0
        for r in rows:
            shots = shots_by_subj.get(r["subject"], [])
            prompt = render(r, shots)
            pick, _ = _mc_pick(model, enc, prompt, [" A", " B", " C", " D"])
            hit = int(pick == r["answer"])
            correct += hit
            per_subj.setdefault(r["subject"], []).append(hit)
        acc = correct / max(1, len(rows))
        return {
            "acc": acc,
            "per_subject": {k: sum(v) / len(v) for k, v in per_subj.items()},
            "n": len(rows), "k_shot": k_shot, "seed": seed,
            "provenance": f"cais/mmlu all/test n={len(rows)} seed={seed} "
                          f"{k_shot}-shot letter-likelihood tokenizer={getattr(enc, 'name', '?')} "
                          f"model_vocab={model.token_embedding.num_embeddings}",
            "claim": "published MMLU rows stay recorded in graded/figures "
                     "(this run measures whatever weights live in this notebook)",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# GSM8K (Cobbe et al. 2021): 8-shot greedy generation, final-number EM
# ---------------------------------------------------------------------------

_GSM8K_SHOTS = (
    ("Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "A: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"),
    ("Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "A: Weng earns 12/60 = 0.2 dollars per minute. Working 50 minutes, she earned 0.2 x 50 = 10 dollars.\n#### 10"),
    ("Q: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. After the first ship it ate, a new ship appeared every two years. How many ships has it eaten after 300 years?",
     "A: Rising once a hundred years means three feasts for a hundred years. There won't be a hundred feasts, but 300 / 100 = 3. If the monster had ate 8 ships in the first 100 years and another 8 ships in the second century, that means it has consumed 3 feasts.\n#### 3"),
    ("Q: Joy can read 8 pages of a book in 20 minutes. How many hours can she read if she has 648 pages she wants to finish after reading a lot?",
     "A: In 20 minutes she reads 8 pages, so in one hour she reads 8 x 3 = 24 pages. So she will finish 648 / 24 = 27 hours.\n#### 27"),
    ("Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
     "A: Betty has half of the money she needs: $50. Her parents gave her 15 + her grandparents twice as much as her parents: 30. So she already has 50 + 15 + 30 = 95 dollars. So she still needs 100 - 95 = 5 dollars.\n#### 5"),
    ("Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "A: He writes 3-page letters twice a week so 3 x 2 = 6 pages a week, twice a week so 6 x 2 = 12 pages. A year has 52 weeks, so he writes 12 x 52 = 624 pages a year.\n#### 624"),
    ("Q: Ken had fifty pencils, and his classmate had ten times more pencils than Ken. Then his math teacher in order to reward him gave Ken some pencils. Ken now has 100 more than his classmate. How many pencils did the teacher give Ken?",
     "A: Ken's classmate had 50 x 10 = 500 pencils. Ken now has 500 + 100 = 600 pencils. So the teacher gave Ken 600 - 50 = 550 pencils.\n#### 550"),
    ("Q: Since 2000, the number of wolves in the park has increased by 20%. If there were 200 wolves in 2000, how many were there 20 years later?",
     "A: The increase was 200 x 20 / 100 = 40. So there will be 200 + 40 = 240 wolves.\n#### 240"),
)

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str) -> Optional[str]:
    nums = _NUM.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def bench_gsm8k(model, enc, n: int = 100, seed: int = 0) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        ds = _hf("openai/gsm8k", "main", "test")
        rng = random.Random(seed)
        rows = list(ds)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        shot = "\n\n".join(f"{q}\n{a}" for q, a in _GSM8K_SHOTS)
        hits, parsed, gen_lens = 0, 0, []
        for r in rows:
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            ans = _greedy_generate(model, enc, shot + "\n\nQ: " + r["question"] + "\nA:",
                                   max_new=256, stop=("\nQ:", "\n\n####", "\n\nQ"))
            gold_n = _last_number(gold)
            gen_n = _last_number(ans)
            if gen_n is not None:
                parsed += 1
            if gold_n is not None and gen_n is not None and abs(float(gen_n) - float(gold_n)) < 1e-6:
                hits += 1
            gen_lens.append(len(enc.encode(ans)))
        return {
            "acc": hits / max(1, len(rows)),
            "answer_parse_rate": parsed / max(1, len(rows)),
            "mean_gen_tokens": sum(gen_lens) / max(1, len(gen_lens)),
            "n": len(rows), "shots": len(_GSM8K_SHOTS), "seed": seed,
            "provenance": f"openai/gsm8k main/test n={len(rows)} seed={seed} "
                          f"{len(_GSM8K_SHOTS)}-shot greedy-EM tokenizer={getattr(enc, 'name', '?')}",
            "claim": "published GSM8K rows stay recorded in graded/figures",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# MBPP (Austin et al. 2021), sanitized: 3-shot, greedy, executed pass@1
# ---------------------------------------------------------------------------

def _exec_tests(code: str, tests: Sequence[str], timeout: int = 10) -> bool:
    prog = code.rstrip() + "\n\n" + "\n".join(t.strip() for t in tests if t.strip()) + "\n"
    d = Path(tempfile.mkdtemp(prefix="mbpp_eval_"))
    prog_file = d / "prog.py"
    prog_file.write_text(prog, encoding="utf-8")
    try:
        p = subprocess.run([sys.executable, "-I", str(prog_file)],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(d))
        return p.returncode == 0
    except Exception:  # noqa: BLE001 - timeout / spawn issues == failure
        return False
    finally:
        try:
            prog_file.unlink()
        except OSError:
            pass


def bench_mbpp(model, enc, n: int = 200, seed: int = 0) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        ds = _pick_repo([("google-research-dataset/mbpp", "sanitized"),
                         ("mbpp", "sanitized")], "test")
        rows = list(ds[0])
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        shots = rows[:3]
        shot_text = "\n\n".join(
            f'"""\n{r["prompt"]}\n"""\n{r["code"]}' for r in shots)
        passed = 0
        ran = 0
        for r in rows[3:]:
            prompt = (shot_text + f'\n\n"""\n{r["prompt"]}\n"""\n')
            gen = _greedy_generate(model, enc, prompt, max_new=256,
                                   stop=('\n\n"""', "\n\n# ", "\n\nassert"))
            gen = gen.strip()
            if gen.startswith('"""'):
                gen = gen[3:]
            ok = _exec_tests(gen, r.get("test_list") or [])
            ran += (ok is not None)
            passed += int(ok)
        total = len(rows) - 3
        return {
            "pass@1": passed / max(1, total),
            "n": total, "shots": 3, "seed": seed,
            "provenance": f"mbpp sanitized/test n={total} seed={seed} 3-shot "
                          f"greedy pass@1 executed in isolated subprocess "
                          f"tokenizer={getattr(enc, 'name', '?')}",
            "claim": "published MBPP rows stay recorded in graded/figures",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# TruthfulQA (Lin et al. 2022): MC1 + MC2 likelihood scoring, full validation
# ---------------------------------------------------------------------------

def bench_truthfulqa(model, enc) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        ds = _hf("truthfulqa/truthful_qa", "multiple_choice", "validation")
        mc1 = mc2 = 0
        rows = list(ds)
        for r in rows:
            t1 = r["mc1_targets"]
            correct = [a for a, l in zip(t1["choices"], t1["labels"]) if l == 1]
            wrong = [a for a, l in zip(t1["choices"], t1["labels"]) if l != 1]
            q = r["question"] + "\nAnswer:"
            ai, _ = _mc_pick(model, enc, q, list(correct) + list(wrong))
            if ai < len(correct):
                mc1 += 1
            t2 = r["mc2_targets"]
            corr2 = [a for a, l in zip(t2["choices"], t2["labels"]) if l == 1]
            wr2 = [a for a, l in zip(t2["choices"], t2["labels"]) if l != 1]
            mc2 += _prob_mass(model, enc, q, corr2, wr2)
        n = len(rows)
        return {
            "mc1": mc1 / max(1, n), "mc2": mc2 / max(1, n), "n": n,
            "provenance": f"truthfulqa/truthful_qa multiple_choice/validation n={n} "
                          f"likelihood mc1+mc2 tokenizer={getattr(enc, 'name', '?')}",
            "claim": "published TruthfulQA rows stay recorded in graded/figures",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# HaluEval (Li et al. 2023), dialogue: pick knowledgeable vs hallucinated reply
# ---------------------------------------------------------------------------

def bench_halieval(model, enc, n: int = 1000, seed: int = 0) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        rows, repo = None, ""
        try:
            ds = _hf("pminervini/HaluEval", "dialogue", "test")
            rows, repo = list(ds), "pminervini/HaluEval dialogue/test"
        except Exception:
            pass
        if rows is None:
            import requests
            url = "https://huggingface.co/datasets/pminervini/HaluEval/resolve/main/dialogue/halu_eval_dialogue.jsonl"
            txt = requests.get(url, timeout=60).text
            rows = [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
            repo = "HaluEval dialogue jsonl (direct fetch)"
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        hits = 0
        skipped = 0
        for r in rows:
            knowledge = r.get("knowledge", "")
            hist = r.get("dialog_history", r.get("history", ""))
            right = r.get("right_answer")
            hall = r.get("hallucinated_answer")
            if not (knowledge and right and hall):
                skipped += 1
                continue
            prompt = (f"Dialogue: {hist}\nKnowledge: {knowledge}\n"
                      f"Which response is factually correct?\nResponse: ")
            ai, _ = _mc_pick(model, enc, prompt, [str(right), str(hall)])
            hits += int(ai == 0)
        used = len(rows) - skipped
        return {
            "acc": hits / max(1, used),
            "n": used, "skipped": skipped, "seed": seed,
            "provenance": f"{repo} n={used} seed={seed} pairwise-likelihood "
                          f"tokenizer={getattr(enc, 'name', '?')}",
            "claim": "published HaluEval rows stay recorded in graded/figures",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# FaithDial (Dziri et al. 2022): faithful vs hallucinated continuation pick
# ---------------------------------------------------------------------------

def bench_faithdial(model, enc, n: int = 1000, seed: int = 0) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        ds = _hf("McGill-NLP/FaithDial", None, "validation")
        rows = list(ds)
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        hits = used = 0
        for r in rows:
            history = " ".join(r.get("history", []) or [])
            good = r.get("response", "")
            bad = r.get("hallucination", "")
            if not (history and good and bad):
                continue
            prompt = f"Dialogue: {history}\nWhich response follows faithfully?\nResponse: "
            ai, _ = _mc_pick(model, enc, prompt, [good, bad])
            hits += int(ai == 0)
            used += 1
        return {
            "acc": hits / max(1, used), "n": used, "n_requested": len(rows),
            "seed": seed,
            "provenance": f"McGill-NLP/FaithDial validation n={used} seed={seed} "
                          f"faithful-vs-hallucinated likelihood "
                          f"tokenizer={getattr(enc, 'name', '?')}",
            "claim": "published FaithDial rows stay recorded in graded/figures",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# OlympiadBench (He et al. 2024): 1-vs-1 likelihood on gold vs perturbed answer
# ---------------------------------------------------------------------------

def bench_olympiad(model, enc, n: int = 200, seed: int = 0) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        ds, repo = _pick_repo([("LLM360/OlympiadBench", None),
                               ("Math-AIGC/OlympiadBench", None)], "train")
        rows = list(ds)
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        hits = used = 0
        per_type: Dict[str, List[int]] = {}
        for r in rows:
            q = str(r.get("question", ""))[:2000]
            gold = str(r.get("final_answer", "")).strip()
            if not q or not gold or gold.lower() in ("none", "nan", ""):
                continue
            m = _last_number(gold)
            if m is None:
                wrong = gold + "0"
            else:
                wrong = str(float(m) + 17).rstrip("0").rstrip(".") or str(float(m) + 17)
            prompt = f"Problem: {q}\n\nGive the final answer.\nAnswer: "
            ai, _ = _mc_pick(model, enc, prompt, [gold, wrong])
            per_type.setdefault(str(r.get("subject", "?")), []).append(int(ai == 0))
            hits += int(ai == 0)
            used += 1
        return {
            "acc": hits / max(1, used),
            "per_type": {k: sum(v) / len(v) for k, v in per_type.items()},
            "n": used, "seed": seed, "repo": repo,
            "provenance": f"{repo} train n={used} seed={seed} gold-vs-perturbed "
                          f"likelihood tokenizer={getattr(enc, 'name', '?')}",
            "claim": "published OlympiadBench rows stay recorded in graded/figures",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# RULER (Hsieh et al. 2024) style NIAH sweep - built from the local corpus,
# capped at the compiled model's single-pass ctx; >ctx lengths are reported
# as skipped-with-reason (long-range coverage runs through the ConvMem runner).
# ---------------------------------------------------------------------------

def bench_ruler(model, enc, ctx_lens: Sequence[int] = (512, 1024, 2048),
                n_per_len: int = 5, seed: int = 0, n_keys: int = 8,
                corpus_text: str = "") -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        if not corpus_text:
            return _err("corpus_text required (pass the notebook corpus)")
        rng = random.Random(seed)
        words = re.findall(r"[A-Za-z]+", corpus_text)
        if len(words) < 4000:
            return _err("corpus too short for RULER builds")
        rows, skipped = [], []
        for L in ctx_lens:
            if L > model.ctx_len:
                skipped.append({"ctx_len": L,
                                "reason": f"exceeds single-pass ctx_len={model.ctx_len}; "
                                          "long-range coverage runs through convmem recursion"})
                continue
            ems = []
            for _ in range(n_per_len):
                vals = {str(rng.randint(10**5, 10**6)): None for _ in range(n_keys)}
                needle_ids, want_key = [], rng.choice(list(vals))
                for k in vals:
                    if k == want_key:
                        continue
                    v = str(rng.randint(10**5, 10**6))
                    vals[k] = v
                    needle_ids.append(f"One of the special magic numbers for key-{k} is: {v}.")
                gold = None
                while gold is None or gold in vals:
                    gold = str(rng.randint(10**5, 10**6))
                vals[want_key] = gold
                needle = f"One of the special magic numbers for key-{want_key} is: {gold}."
                needles = [needle] + needle_ids
                rng.shuffle(needles)
                fill = words
                hay: List[str] = []
                budget = L - sum(len(enc.encode(p)) for p in needles + [
                    f"What is the special magic number for key-{want_key} mentioned in the provided text?"])
                pos = 0
                while sum(len(enc.encode(w)) for w in hay) < budget and pos < len(fill) * 50:
                    hay.append(fill[pos % len(fill)])
                    pos += 1
                filler = " ".join(hay)
                chunks = [filler[:max(1, len(filler) // (n_keys + 1))]] * (n_keys + 1)
                ctx_parts: List[str] = []
                ni = 0
                for j, c in enumerate(chunks):
                    ctx_parts.append(c)
                    if ni < len(needles):
                        ctx_parts.append(" " + needles[ni])
                        ni += 1
                question = f"What is the special magic number for key-{want_key} mentioned in the provided text?"
                prompt = " ".join(ctx_parts) + "\n\n" + question + " Answer:"
                gen = _greedy_generate(model, enc, prompt, max_new=12, stop=("\n",))
                m = _NUM.findall(gen.replace(",", ""))
                ems.append(int(bool(m) and m[-1] == gold))
            rows.append({"ctx_len": L, "em": sum(ems) / max(1, len(ems)),
                         "n": len(ems)})
        return {
            "rows": rows,
            "skipped": skipped,
            "n_per_len": n_per_len, "n_keys": n_keys, "seed": seed,
            "provenance": f"RULER-style NIAH over local corpus n_per_len={n_per_len} "
                          f"seed={seed} greedy-EM tokenizer={getattr(enc, 'name', '?')} "
                          f"single-pass lens only",
            "claim": "published RULER rows stay recorded in graded/figures; "
                     "28k-896k coverage runs through the ConvMem hierarchy runner",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)
# ---------------------------------------------------------------------------
# s38 pragmatic families (arXiv:2610.21204): the think-dial reading probe on
# IMPPRES / NOPE / LUDWIG / Fig-QA / PUB / Circa - the benchmark families the
# paper's Figures 2/5/6/8 run on.
#
# Shared protocol (the paper's section 2, realized):
#   - every item is a FORCED-choice question about what the speaker MEANT
#     ("meaning" framing); IMPPRES additionally carries the true/false
#     "logic mode" framing the paper uses as the literal-gold comparison;
#   - options are shuffled per trial, with the SAME shuffled order shared
#     across the two think arms of an item (so the arm delta is within-item);
#   - two think arms on the same weights: "off" (direct answer) and "capped"
#     (a greedy reasoning fill of `budget` tokens inserted between the item
#     body and the options). Declared adaptation: the paper switches a
#     reasoning model's thinking off / stops it after B tokens; these
#     compiled models have no thinking switch, so the capped arm is the
#     greedy fill stand-in - the dial mechanics (same weights, budget knob)
#     are what transfer;
#   - scoring: likelihood forced-choice over the options (the paper's
#     headline is likewise forced-choice; free-form was dropped there as
#     unreliable to code);
#   - readouts per family and arm: accuracy against the dataset's own gold
#     and, where the dataset carries both a literal and a pragmatic NLI
#     label (IMPPRES, NOPE), acc_log vs acc_prag with the committed
#     pragmatic reading R = acc_prag - the paper's dR anchors are recorded
#     claims (Fig-QA +10.7 pp, PUB +10.1 pp, Circa +17.5 pp, NOPE +9.9 pp,
#     IMPPRES presupposition +22.2 pp, some -28.2 pp), never simulated here.
# ---------------------------------------------------------------------------

_LETTERS = "ABCDEFGH"


def _think_fill(model, enc, prefix: str, budget: int) -> str:
    """Greedy reasoning fill of `budget` tokens appended to `prefix`
    (the capped-thinking arm, slid under the compiled ctx)."""
    if int(budget) <= 0:
        return prefix
    ids = enc.encode(prefix)
    ctx = ids[-min(len(ids), model.ctx_len):]
    gen: List[int] = []
    for _ in range(int(budget)):
        with torch.no_grad():
            x = torch.tensor([ctx], device=_dev(model))
            logits = model(x)[0]
            nxt = int(torch.argmax(logits[-1]))
        gen.append(nxt)
        ctx = (ctx + [nxt])[-model.ctx_len:]
    return prefix + " " + enc.decode(gen).strip()


def _options_block(opts: Sequence[str]) -> Tuple[List[str], str]:
    letters = list(_LETTERS[: len(opts)])
    tail = "\nOptions:\n" + "\n".join(
        f"{l}. {o}" for l, o in zip(letters, opts))
    tail += "\nRespond with only the letter of the best answer.\nAnswer:"
    return letters, tail


def _think_arms(model, enc, body: str, opts: Sequence[str], budget: int
                ) -> Tuple[int, int]:
    """The think dial for ONE item: 'off' answers directly; 'capped' gets a
    greedy reasoning fill of `budget` tokens before the options. Returns
    (pick_off, pick_capped) answer indices into opts."""
    letters, tail = _options_block(opts)
    cands = [" " + l for l in letters]
    off, _ = _mc_pick(model, enc, body + tail, cands)
    if int(budget) > 0:
        filled = _think_fill(model, enc, body, int(budget))
        cap, _ = _mc_pick(model, enc, filled + tail, cands)
    else:
        cap = off
    return off, cap


def _nli_int(v: Any) -> int:
    """Normalize an NLI label to entailment=0 / neutral=1 / contradiction=2."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("entail"):
            return 0
        if s.startswith("neutr"):
            return 1
        if s.startswith("contrad"):
            return 2
        return -1
    i = int(v)
    return i if i in (0, 1, 2) else -1


def _shuffle_pair(rng: random.Random, yes: str, no: str) -> Tuple[List[str], int]:
    """Per-trial 2-option shuffle; returns (shuffled options, yes-index)."""
    idx = rng.randrange(2)
    return ([yes, no] if idx == 0 else [no, yes]), idx


# ---------------------------------------------------------------------------
# IMPPRES (Jeretic et al. 2020): scalar implicature + presupposition items
# under the dual literal/pragmatic NLI labels.
# ---------------------------------------------------------------------------

_IMPPRES_IMPLICATURE = ("implicature_connectives", "implicature_gradable_adjective",
                        "implicature_gradable_verb", "implicature_modals",
                        "implicature_quantifiers", "implicature_numerals_10_100",
                        "implicature_numerals_2_3")
_IMPPRES_PRESUP = ("presupposition_all_n_presupposition", "presupposition_both_presupposition",
                   "presupposition_change_of_state", "presupposition_cleft_existence",
                   "presupposition_cleft_uniqueness", "presupposition_only_presupposition",
                   "presupposition_possessed_definites_existence",
                   "presupposition_possessed_definites_uniqueness",
                   "presupposition_question_presupposition")


def bench_imppres(model, enc, n: int = 240, seed: int = 38, budget: int = 32,
                  families: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        fams = list(families) if families else list(_IMPPRES_IMPLICATURE) + \
            list(_IMPPRES_PRESUP)
        rng = random.Random(seed)
        per_fam: Dict[str, Dict[str, Any]] = {}
        skipped: List[Dict[str, str]] = []
        n_used = 0
        for conf in fams:
            try:
                split = conf.replace("implicature_", "").replace("presupposition_", "")
                ds = _pick_repo([("facebook/imppres", conf)], split)
            except Exception as e:  # noqa: BLE001 - family not measured here
                skipped.append({"family": conf, "reason": str(e)[:200]})
                continue
            rows = list(ds[0])
            if len(rows) > n:
                rows = rng.sample(rows, n)
            hits = {"log_off": 0, "log_cap": 0, "prag_off": 0, "prag_cap": 0,
                    "logic_off": 0, "logic_cap": 0, "n": 0}
            for r in rows:
                p = str(r.get("premise", "")).strip()
                h = str(r.get("hypothesis", "")).strip()
                g_log = _nli_int(r.get("gold_label_log", -1))
                g_prag = _nli_int(r.get("gold_label_prag", -1))
                if not p or not h or g_log < 0:
                    continue
                body = (f'Speaker: "{p}"\nWould the listener conclude '
                        f'that the speaker meant that "{h}"?\n')
                (oy, i_y) = _shuffle_pair(rng, "yes", "no")
                off, cap = _think_arms(model, enc, body, oy, budget)
                off_is_yes, cap_is_yes = off == i_y, cap == i_y
                gold_log_yes = g_log == 0
                hits["log_off"] += int(off_is_yes == gold_log_yes)
                hits["log_cap"] += int(cap_is_yes == gold_log_yes)
                hits["prag_off"] += int(off_is_yes == (g_prag == 0))
                hits["prag_cap"] += int(cap_is_yes == (g_prag == 0))
                lbody = (f'Sentence 1: "{p}"\nSentence 2: "{h}"\n'
                         "Is Sentence 2 necessarily true given Sentence 1?\n")
                ly, li_y = _shuffle_pair(rng, "yes", "no")
                loff, lcap = _think_arms(model, enc, lbody, ly, budget)
                hits["logic_off"] += int((loff == li_y) == gold_log_yes)
                hits["logic_cap"] += int((lcap == li_y) == gold_log_yes)
                hits["n"] += 1
            if hits["n"]:
                m = hits["n"]
                per_fam[conf] = {
                    "n": m,
                    "acc_log": {"off": round(hits["log_off"] / m, 4),
                                "capped": round(hits["log_cap"] / m, 4)},
                    "acc_prag": {"off": round(hits["prag_off"] / m, 4),
                                 "capped": round(hits["prag_cap"] / m, 4)},
                    "acc_logic": {"off": round(hits["logic_off"] / m, 4),
                                  "capped": round(hits["logic_cap"] / m, 4)},
                    "R_prag_log_gap_pp": {
                        "off": round(100.0 * (hits["prag_off"] - hits["log_off"]) / m, 2),
                        "capped": round(100.0 * (hits["prag_cap"] - hits["log_cap"]) / m, 2)},
                }
                n_used += m
        if not per_fam:
            return _err("imppres: no family accessible (all errored/skipped)")
        return {
            "families": per_fam,
            "skipped_families": skipped,
            "n": n_used, "n_per_family": n, "seed": seed, "budget": budget,
            "provenance": f"facebook/imppres per-config test splits n<= {n}/fam "
                          f"seed={seed} budget={budget} likelihood forced-choice, "
                          f"options shuffled per trial; title=think dial off/capped "
                          f"tokenizer={getattr(enc, 'name', '?')}",
            "claim": "paper anchors recorded, not simulated: some -28.2/or -36.4 pp "
                     "@Qwen3-8B; presupposition +22.2 pp",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# NOPE (Parrish et al. 2021): naturally-occurring presuppositions (CoNLL),
# direct GitHub jsonl (no HF mirror; the corpus carries no licence and is
# used for research only, per the paper's A.2 note).
# ---------------------------------------------------------------------------

_NOPE_URLS = ("https://raw.githubusercontent.com/nyu-mll/nope/main/"
              "annotated_corpus/nli_corpus.main.jsonl",
              "https://github.com/nyu-mll/nope/raw/main/"
              "annotated_corpus/nli_corpus.main.jsonl")


def _nope_rows() -> List[Dict[str, Any]]:
    """Main NOPE corpus (2,386 NLI items), cached under data/external/nope."""
    cache = DATA / "nope" / "nli_corpus.main.jsonl"
    if cache.exists():
        return [json.loads(ln) for ln in cache.read_text(
            encoding="utf-8").splitlines() if ln.strip()]
    import requests
    last: Optional[Exception] = None
    for url in _NOPE_URLS:
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            rows = [json.loads(ln) for ln in resp.text.splitlines() if ln.strip()]
            if rows:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(resp.text, encoding="utf-8")
                return rows
        except Exception as e:  # noqa: BLE001 - fail-soft contract
            last = e
    raise IOError(f"nope corpus fetch failed: {last}")


def bench_nope(model, enc, n: int = 400, seed: int = 38, budget: int = 32) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        rows_all = _nope_rows()
        rng = random.Random(seed)
        rows = list(rows_all)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        per_trig: Dict[str, Dict[str, Any]] = {}
        hits_acc = {"off": 0, "cap": 0}
        n_used = 0
        for r in rows:
            p = str(r.get("premise", "")).strip().replace("\n", " ")
            h = str(r.get("hypothesis", "")).strip().replace("\n", " ")
            lab = str(r.get("label", "")).strip().upper()[:1]
            g_log = {"E": 0, "N": 1, "C": 2}.get(lab, -1)
            if not p or not h or g_log < 0:
                continue
            body = (f'Speaker: "{p}"\nIs the speaker thereby committed to '
                    f'this being true: "{h}"?\n')
            oy, i_y = _shuffle_pair(rng, "yes", "no")
            off, cap = _think_arms(model, enc, body, oy, budget)
            gold_yes_off = g_log == 0
            hit_o = (off == i_y) == gold_yes_off
            hit_c = (cap == i_y) == gold_yes_off
            hits_acc["off"] += int(hit_o)
            hits_acc["cap"] += int(hit_c)
            md = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
            trig = str(md.get("trigger_type", "?"))
            if str(md.get("type", "original")) == "negated":
                trig += "/negated"
            d = per_trig.setdefault(
                trig, {"n": 0, "acc_off": 0, "acc_capped": 0})
            d["n"] += 1
            d["acc_off"] += int(hit_o)
            d["acc_capped"] += int(hit_c)
            n_used += 1
        if not n_used:
            return _err("nope: no scored rows")
        return {
            "acc": {"off": round(hits_acc["off"] / n_used, 4),
                    "capped": round(hits_acc["cap"] / n_used, 4)},
            "delta_pp": round(100.0 * (hits_acc["cap"] - hits_acc["off"]) / n_used, 2),
            "per_trigger": {k: {"n": v["n"],
                                "acc_off": round(v["acc_off"] / v["n"], 4),
                                "acc_capped": round(v["acc_capped"] / v["n"], 4)}
                            for k, v in per_trig.items()},
            "n": n_used, "corpus_size": len(rows_all), "seed": seed,
            "budget": budget,
            "provenance": "nyu-mll/nope nli_corpus.main.jsonl (direct GitHub "
                          "fetch, research use) n=%d corpus=%d seed=%d "
                          "budget=%d projection yes/no forced-choice, options "
                          "shuffled per trial tokenizer=%s" % (
                              n_used, len(rows_all), seed, budget,
                              getattr(enc, "name", "?")),
            "claim": "paper anchor recorded, not simulated: NOPE +9.9 pp "
                     "(accuracy, Qwen3-8B)",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# LUDWIG (Ruis et al. 2023): naturally-occurring conversational implicatures,
# binary yes/no implicature resolution 0-shot.
# ---------------------------------------------------------------------------

def bench_ludwig(model, enc, n: int = 718, seed: int = 38, budget: int = 32,
                 k_shots: int = 0) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        conf = {0: "0-shot", 1: "1-shot", 5: "5-shot", 10: "10-shot",
                15: "15-shot", 30: "30-shot"}.get(int(k_shots), "0-shot")
        split, ds_rep = "", None
        last: Optional[Exception] = None
        for sp in ("test", "validation"):
            try:
                ds_rep = _hf("UCL-DARK/ludwig", conf, sp)
                split = sp
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if ds_rep is None:
            raise IOError(f"UCL-DARK/ludwig/{conf} failed: {last}")
        rows = list(ds_rep)
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        hits = {"off": 0, "cap": 0}
        n_used = 0
        for r in rows:
            utt = str(r.get("utterance", "")).strip()
            resp = str(r.get("response", "")).strip()
            imp = str(r.get("implicature", "")).strip().lower()
            if not (utt and resp and imp in ("yes", "no")):
                continue
            shots_txt = ""
            if int(k_shots) > 0:
                parts = []
                for pr in (r.get("prompts") or [])[: int(k_shots)]:
                    pu = str(pr.get("utterance", "")).strip()
                    prs = str(pr.get("response", "")).strip()
                    pi = str(pr.get("implicature", "")).strip().lower()
                    if pu and prs and pi in ("yes", "no"):
                        parts.append(f'Speaker A says: "{pu}"\n'
                                     f'Speaker B responds: "{prs}"\n'
                                     f'Does B\'s response mean "{pi}"? - {pi}\n')
                if parts:
                    shots_txt = "\n\n".join(parts) + "\n\n"
            body = (f'Speaker A says: "{utt}"\n'
                    f'Speaker B responds: "{resp}"\n'
                    'Does B\'s answer mean "yes" to A\'s question?\n')
            oy, i_y = _shuffle_pair(rng, "yes", "no")
            off, cap = _think_arms(model, enc, shots_txt + body, oy, budget)
            gold_yes = (imp == "yes")
            hits["off"] += int((off == i_y) == gold_yes)
            hits["cap"] += int((cap == i_y) == gold_yes)
            n_used += 1
        if not n_used:
            return _err("ludwig: no scored rows")
        return {
            "acc": {"off": round(hits["off"] / n_used, 4),
                    "capped": round(hits["cap"] / n_used, 4)},
            "delta_pp": round(100.0 * (hits["cap"] - hits["off"]) / n_used, 2),
            "n": n_used, "k_shots": int(k_shots), "split": split, "seed": seed,
            "budget": budget,
            "provenance": f"UCL-DARK/ludwig {conf}/{split} n={n_used} seed={seed} "
                          f"budget={budget} binary yes/no forced-choice, options "
                          f"shuffled per trial tokenizer={getattr(enc, 'name', '?')}",
            "claim": "published LUDWIG rows (Ruis et al. 2023 accuracy 79.7/84.0/82.0 "
                     "0-shot @NT/512/NAT) stay recorded in graded/figures",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# Fig-QA (Liu et al. 2022): figurative language Winograd pairs - the correct
# option is the figurative-complementary one; accuracy here IS the pragmatic
# reading share R of the probe.
# ---------------------------------------------------------------------------

def bench_fig_qa(model, enc, n: int = 1146, seed: int = 38, budget: int = 32) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        split, ds_rep = "", None
        last: Optional[Exception] = None
        for sp in ("test", "validation", "train"):
            try:
                ds_rep, split = _pick_repo([("nightingal3/fig-qa", None)], sp), sp
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if ds_rep is None:
            raise IOError(f"nightingal3/fig-qa failed: {last}")
        rows = list(ds_rep[0])
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        hits = {"off": 0, "cap": 0}
        n_used = 0
        for r in rows:
            sp_txt = str(r.get("startphrase", "")).strip()
            e1 = str(r.get("ending1", "")).strip()
            e2 = str(r.get("ending2", "")).strip()
            lab = r.get("labels", None)
            if not (sp_txt and e1 and e2) or lab is None or int(lab) not in (0, 1):
                continue
            ll = int(lab)
            opts = [e1, e2] if rng.randrange(2) == 0 else [e2, e1]
            want = opts.index([e1, e2][ll])
            body = f'"{sp_txt}"\nWhich continuation is the better fit?\n'
            off, cap = _think_arms(model, enc, body, opts, budget)
            hits["off"] += int(off == want)
            hits["cap"] += int(cap == want)
            n_used += 1
        if not n_used:
            return _err("fig_qa: no scored rows")
        return {
            "acc_R": {"off": round(hits["off"] / n_used, 4),
                      "capped": round(hits["cap"] / n_used, 4)},
            "delta_pp": round(100.0 * (hits["cap"] - hits["off"]) / n_used, 2),
            "n": n_used, "split": split, "seed": seed, "budget": budget,
            "provenance": f"nightingal3/fig-qa {split} n={n_used} seed={seed} "
                          f"budget={budget} 2-option likelihood forced-choice "
                          f"(Winograd pairs, figurative = correct) options "
                          f"shuffled per trial tokenizer={getattr(enc, 'name', '?')}",
            "claim": "paper anchor recorded, not simulated: Fig-QA +10.7 pp",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# Circa (Louis et al. 2020): indirect answers to polar questions - accuracy
# over the dataset's own 5-label interpretation space (goldstandard2).
# ---------------------------------------------------------------------------

_CIRCA_LABELS = ("Yes", "No", "In the middle, neither yes nor no",
                 "Yes, subject to some conditions", "Other")


def bench_circa(model, enc, n: int = 1000, seed: int = 38, budget: int = 32) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        split, ds_rep = "", None
        last: Optional[Exception] = None
        for sp in ("test", "train"):
            try:
                ds_rep, split = _pick_repo(
                    [("google-research-datasets/circa", None)], sp), sp
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if ds_rep is None:
            raise IOError(f"circa fetch failed: {last}")
        rows = list(ds_rep[0])
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        hits = {"off": 0, "cap": 0}
        per_gold: Dict[str, List[int]] = {}
        n_used = skipped_gold = 0
        for r in rows:
            ctx = str(r.get("context", "")).strip()
            qx = str(r.get("question-X", "")).strip()
            ay = str(r.get("answer-Y", "")).strip()
            g2 = r.get("goldstandard2", None)
            if g2 is None or int(g2) < 0:
                skipped_gold += 1
                continue
            if not (ctx and qx and ay):
                skipped_gold += 1
                continue
            gold = _CIRCA_LABELS[int(g2) % len(_CIRCA_LABELS)]
            opts = list(_CIRCA_LABELS)
            rng.shuffle(opts)
            want = opts.index(gold)
            body = (f"Context: {ctx}\nX asks: {qx}\nY answers: {ay}\n"
                    "What does Y's answer mean?\n")
            off, cap = _think_arms(model, enc, body, opts, budget)
            ho = int(off == want)
            hc = int(cap == want)
            hits["off"] += ho
            hits["cap"] += hc
            per_gold.setdefault(gold, [0, 0, 0])
            per_gold[gold][0] += 1
            per_gold[gold][1] += ho
            per_gold[gold][2] += hc
            n_used += 1
        if not n_used:
            return _err("circa: no scored rows")
        return {
            "acc": {"off": round(hits["off"] / n_used, 4),
                    "capped": round(hits["cap"] / n_used, 4)},
            "delta_pp": round(100.0 * (hits["cap"] - hits["off"]) / n_used, 2),
            "per_gold_label": {k: {"n": v[0],
                                   "acc_off": round(v[1] / v[0], 4),
                                   "acc_capped": round(v[2] / v[0], 4)}
                               for k, v in per_gold.items()},
            "n": n_used, "skipped_no_consensus": skipped_gold,
            "split": split, "seed": seed, "budget": budget,
            "provenance": f"google-research-datasets/circa {split} n={n_used} "
                          f"seed={seed} budget={budget} MC over the dataset's own "
                          f"label space (goldstandard2; -1 consensus rows skipped), "
                          f"options shuffled per trial tokenizer={getattr(enc, 'name', '?')}",
            "claim": "paper anchor recorded, not simulated: Circa +17.5 pp",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------------------------------------------------------------------------
# PUB (Sravanthi et al. 2024): 14 pragmatic MCQA tasks over implicature,
# presupposition, reference, deixis - MC with the dataset's own options.
# ---------------------------------------------------------------------------

def bench_pub(model, enc, n: int = 400, seed: int = 38, budget: int = 32) -> Dict[str, Any]:
    try:
        if enc is None:
            return _err("no encoder matches the compiled vocab")
        ds = _pick_repo([("cfilt/PUB", None)], "train")
        rows = list(ds[0])
        rng = random.Random(seed)
        if len(rows) > n:
            rows = rng.sample(rows, n)
        hits = {"off": 0, "cap": 0}
        per_sz: Dict[str, Dict[str, int]] = {}
        n_used = skipped = 0
        for r in rows:
            pretext = str(r.get("pretext", "")).strip()
            opts = [str(o) for o in (r.get("options") or [])]
            gold = str(r.get("correct answer", "")).strip()
            if not pretext or len(opts) > len(_LETTERS) or gold not in opts:
                skipped += 1
                continue
            rng.shuffle(opts)
            want = opts.index(gold)
            body = pretext + "\nChoose the best option.\n"
            off, cap = _think_arms(model, enc, body, opts, budget)
            ho = int(off == want)
            hc = int(cap == want)
            hits["off"] += ho
            hits["cap"] += hc
            key = "n_options_%d" % len(opts)
            d = per_sz.setdefault(key, {"n": 0, "off": 0, "capped": 0})
            d["n"] += 1
            d["off"] += ho
            d["capped"] += hc
            n_used += 1
        if not n_used:
            return _err("pub: no scored rows")
        return {
            "acc": {"off": round(hits["off"] / n_used, 4),
                    "capped": round(hits["cap"] / n_used, 4)},
            "delta_pp": round(100.0 * (hits["cap"] - hits["off"]) / n_used, 2),
            "per_n_options": {k: {"n": v["n"],
                                  "acc_off": round(v["off"] / v["n"], 4),
                                  "acc_capped": round(v["capped"] / v["n"], 4)}
                              for k, v in per_sz.items()},
            "n": n_used, "skipped": skipped, "seed": seed, "budget": budget,
            "provenance": f"cfilt/PUB train n={n_used} seed={seed} budget={budget} "
                          f"MC with the dataset's own options, shuffled per trial "
                          f"tokenizer={getattr(enc, 'name', '?')}",
            "claim": "paper anchor recorded, not simulated: PUB (six tasks) "
                     "+10.1 pp @Qwen3-8B",
        }
    except Exception as e:  # noqa: BLE001
        return _err(e)
