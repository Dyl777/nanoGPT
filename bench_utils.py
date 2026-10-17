"""Shared inputs + reporting for the benchmark call sites in nanoGPT_flat.ipynb.optll.

Imported by the per-section call-site cells. Everything reads only what the
notebook already loaded (`model`, `text`/`encode`/`data`, `get_batch`,
`int_to_token`) - it never builds a second model and never loads a dataset.

Notebook globals are not visible from an imported module, so the call sites pass
their own namespace via `ns=globals()`. When `ns` is omitted we fall back to
`builtins.__dict__`, which is correct for a plain script.

`bench_report` is why a failing call site can never take the notebook down: it
catches everything and reports OK or SKIP with the reason.
"""
import time


def _ns(ns):
    return ns if ns is not None else __builtins__["__dict__"]


def bench_ids(ns=None, n=2048):
    """Token ids from the already-loaded corpus (no file/dataset access)."""
    g = _ns(ns)
    data = g.get("data")
    if data is not None:
        return [int(v) for v in data[:n].tolist()]
    enc, txt = g.get("encode"), g.get("text")
    if enc is not None and txt is not None:
        return [int(v) for v in enc(txt[:n])]
    raise RuntimeError("no token stream in this notebook run")


def bench_pairs_fn(ns=None, n=160, seed=3):
    """Answer/continuation pairs for the LayerMix-style hallucination probes."""
    import torch as _t
    g = _ns(ns)
    _t.manual_seed(seed)
    txt = g.get("text") or ""
    m = g.get("model")
    ctx = min(getattr(m, "ctx_len", 64), 64)
    span = max(8, ctx // 2)
    if not txt:
        return []
    step = max(1, len(txt) // n)
    out = []
    for i in range(n):
        s = (i * step) % max(1, len(txt) - span * 2)
        out.append((txt[s:s + span], txt[s + span:s + span * 2]))
    return out


def bench_tasks(ns=None, n=12):
    """Small QA-style items ({"q","a"}) for the early-exit / prefix / tasco runs."""
    g = _ns(ns)
    txt = g.get("text") or ""
    if not txt:
        return []
    step = max(1, len(txt) // n)
    items = []
    for i in range(n):
        s = (i * step) % max(1, len(txt) - 80)
        items.append({"q": txt[s:s + 40], "a": txt[s + 40:s + 72]})
    return items


def bench_niah(ns=None, n_needles=4, n_distract=20):
    """Needles / distractors / queries for the convolutional-memory recall run."""
    g = _ns(ns)
    txt = g.get("text") or ""
    if not txt:
        return [], [], []
    total = n_needles + n_distract
    step = max(1, len(txt) // (total + 1))
    facts = []
    for i in range(total):
        s = (i * step) % max(1, len(txt) - 60)
        facts.append(txt[s:s + 40])
    needles = facts[:n_needles]
    distractors = facts[n_needles:]
    queries = [{"needle": nd, "query": nd[:12]} for nd in needles]
    return needles, distractors, queries


def _summ(res):
    if not isinstance(res, dict):
        return ""
    measurements = res.get("measurements")
    if isinstance(measurements, dict):
        return "%d measurements" % len(measurements)
    return ", ".join(list(res.keys())[:4])


def bench_report(name, fn, **kw):
    """Call one benchmark runner; never raise into the notebook.

    Returns a status record so the notebook can print an end-of-run summary.
    """
    t0 = time.time()
    try:
        res = fn(**kw)
    except Exception as exc:  # noqa: BLE001 - reporting, not swallowing
        reason = "%s: %s" % (type(exc).__name__, str(exc)[:150])
        print("%-26s SKIP  %s" % (name, reason))
        return {"name": name, "status": "skipped", "reason": reason}
    print("%-26s OK    %6.1fs  %s" % (name, time.time() - t0, _summ(res)))
    return {"name": name, "status": "ok", "seconds": round(time.time() - t0, 2)}