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
