"""Run Graded recreations without compiling OPTAL (same functions the .optll files call)."""

from __future__ import annotations

import argparse
import json

from graded.arxiv_llm_lib import run_paper_1706, run_paper_1810, run_paper_2608, run_paper_27963


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--papers", default="1,2", help="Comma list of paper indices (1=Attention, 2=BERT, 3=LayerMix, 4=SABER)")
    args = p.parse_args()
    wanted = {int(x.strip()) for x in args.papers.split(",") if x.strip()}
    all_results = {}
    if 1 in wanted:
        all_results["1706.03762"] = run_paper_1706()
    if 2 in wanted:
        all_results["1810.04805"] = run_paper_1810()
    if 3 in wanted:
        all_results["2608.28930"] = run_paper_2608()
    if 4 in wanted:
        all_results["2608.27963"] = run_paper_27963()
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
