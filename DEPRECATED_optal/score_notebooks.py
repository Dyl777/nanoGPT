import json, io, contextlib, pathlib

nbdir = pathlib.Path(r"C:\Users\AMBE\Desktop\Engineering\nanoGPT\optal\notebooks")
for nb in sorted(nbdir.glob("*.ipynb")):
    data = json.loads(nb.read_text(encoding="utf-8"))
    src = "\n".join(data["cells"][0]["source"])
    g = {"__name__": "__main__"}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<optal>", "exec"), g)
        if "main" in g:
            g["main"]()
    except Exception as e:
        print(f"{nb.name:34} ERROR {e}")
        continue
    out = [l for l in buf.getvalue().splitlines() if l.startswith("__OPTAL_RESULT__=")]
    if out:
        res = json.loads(out[-1].split("=", 1)[1])
        print(f"{nb.name:34} status={res.get('status')} cosine_sim={res.get('metrics', {}).get('cosine_sim')}")
