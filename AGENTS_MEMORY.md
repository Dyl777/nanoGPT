# Agent memory — mandatory operating rules

These are corrections from the user. Read them before every answer and apply
them to every change in this repository.

## Model scale and vocabulary

- Do **not** call the model a "char model" or "char-model scale".
- `Vocab` and `Scale` are separate axes.
  - `Vocab::[gpt2bpe, char64, word1k, ...]`
  - `Scale::[gpt2_124m, tiny64, small256, faithful512]`
  - `Context::[ctx1024, ctx64, ...]`
- If a section uses char-level *data* by choice, say that; never claim the
  model itself is char-only.

## OPTAL implementation rules

- An `@Impl` must contain the actual implementation selected by the path,
  not `SOMETHING_KIND = "value"` followed by a runner branch.
- Define shared classes/functions **before** the gates that use them.
- Do not duplicate classes across sections. One implementation per concept;
  reuse the substrate or the compiled axis.
- A new axis must change the model/computation, not just a prompt or dataset
  condition. Prompt cues, seeds, datasets and attribution methods are runner
  parameters, not axes.
- Do not add a fake axis to satisfy a checkbox. If the code is not real,
  remove the axis instead.

## Section placement

- Place each paper by **topic and surrounding sections**, not by section
  number and not simply after the previous paper.
- Read the sections before and after the insertion point.
- Theory/context-reasoning work belongs with reasoning measurement; memory
  work with memory; quantization with quantization.
- Do not leave a new section as a chronological append at the end unless that
  really is its topic block.

## Benchmarks

- Benchmark **execution** belongs in the central benchmark manifest
  (`BENCH_SPECS` + `defer_call`, run by the final manifest), not scattered
  through paper sections.
- A section may call `live_call` directly only if it genuinely needs the
  result immediately for a branch, follow-up metric, or in-cell plot.
- If direct execution is used, do not also register the same runner.
- Implement the paper's named baselines and benchmarks; do not stop at
  anchors when the code is implementable.
- Never fabricate numbers. Record `SKIP` with the dataset/checkpoint reason.

## Verification

- Run `optalgo validate` after edits; 0 errors is the bar.
- Compile changed Python modules.
- Do not claim an end-to-end run when only validation/static checks were done.
- Do not modify `graded/`.
- Do not use git commands.
- If something is not implemented, say so plainly; do not hide it behind
  "declared limits" when the user asked for implementation.

## User communication

- No excuses, no repeated mischaracterizations.
- When corrected, fix the code, not just the prose.
- Read the existing implementation patterns before adding new ones.

## Implementation verification and replication fraud checks

- For every paper, first read its official code/repository when available and
  compare the implementation against the paper's equations.
- Check the repository's implementation and open issues for known bugs,
  deviations, missing pieces, or mismatches; record them.
- Deep-research the web when a replication looks fraudulent, inconsistent, or
  too-good-to-be-true; verify claims, numbers, and provenance.
- In the notebook section, explicitly indicate what was checked (repository
  files, issues, web sources), any mismatch found, and the correction made.
- Do not silently reproduce paper claims or repository code; document the
  verification in the section when adding it to the notebook.
- Do not change any other rule in this file.
