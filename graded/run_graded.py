"""Run Graded recreations without compiling OPTAL (same functions the .optll files call)."""

from __future__ import annotations

import argparse
import json

from graded.arxiv_llm_lib import run_paper_1706, run_paper_1810, run_paper_2608, run_paper_27963, run_paper_2609, run_paper_10441, run_paper_10305, run_paper_09883, run_paper_11393


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--papers", default="1,2", help="Comma list of paper indices (1=Attention, 2=BERT, 3=LayerMix, 4=SABER, 5=Grokking, 6=ConvMem, 7=RiLM, 8=WRP, 9=TASCO)")
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
    if 5 in wanted:
        all_results["2609.10657"] = run_paper_2609()
    if 6 in wanted:
        all_results["2609.10441"] = run_paper_10441()
    if 7 in wanted:
        all_results["2609.10305"] = run_paper_10305()
    if 8 in wanted:
        all_results["2609.09883"] = run_paper_09883()
    if 9 in wanted:
        all_results["2609.11393"] = run_paper_11393()
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
