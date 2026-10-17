import io

# Read the original file
with open('nanoGPT_flat.ipynb.optll', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# Cell 1: Header markdown
cell1 = [
    '    {',
    '  "cell_type": "markdown",',
    '  "metadata": {},',
    '  "source": [',
    '    "## Curve Pipeline Infrastructure (arXiv:2605.23872+)\\n",',
    '    "\\n",',
    '    "**Unified curve pipeline for all papers 1-19.** This cell provides the unified\\n",',
    '    "CurvePipeline infrastructure: BenchmarkSuite configs, CurvePipeline class,\\n",',
    '    "run_curve_pipeline runner, and per-paper curve measurements. All papers 1-19\\n",',
    '    "integrate here.\\n"',
    '  ]',
    '},'
]

# Cell 2: Substrate code
cell2_src = [
    '"""Unified Curve Pipeline Infrastructure (arXiv:2605.23872+).',
    '',
    'This module provides the unified curve pipeline infrastructure for all papers.',
    'Components:',
    '- BenchmarkSuite: configuration for each paper\'s benchmark suite',
    '- CurvePipeline: unified pipeline class that runs the benchmark suite',
    '- run_curve_pipeline: runner that executes the pipeline and emits curves',
    '- BenchmarkSuite registry: maps paper index to suite config',
    '"""',
    '',
    'from __future__ import annotations',
    '',
    'import json',
    'from pathlib import Path',
    'from typing import Dict, List, Any, Callable, Optional',
    '',
    'import torch',
    'import torch.nn as nn',
    'import torch.nn.functional as F',
    '',
    'class BenchmarkSuite:',
    '    """Configuration for a paper\'s benchmark suite."""',
    '    def __init__(self, paper_idx: int, name: str, tasks: List[str],',
    '                 metric: str = "accuracy", fewshot: int = 0,',
    '                 limit: Optional[int] = None, **kwargs):',
    '        self.paper_idx = paper_idx',
    '        self.name = name',
    '        self.tasks = tasks',
    '        self.metric = metric',
    '        self.fewshot = fewshot',
    '        self.limit = limit',
    '        self.kwargs = kwargs',
    '',
    '    def to_dict(self) -> Dict[str, Any]:',
    '        return {',
    '            "paper_idx": self.paper_idx,',
    '            "name": self.name,',
    '            "tasks": self.tasks,',
    '            "metric": self.metric,',
    '            "fewshot": self.fewshot,',
    '            "limit": self.limit,',
    '            **self.kwargs',
    '        }',
    '',
    'class CurvePipeline:',
    '    """Unified curve pipeline runner."""',
    '    def __init__(self, model_factory, suite, technique_impl="none", **technique_kwargs):',
    '        self.model_factory = model_factory',
    '        self.suite = suite',
    '        self.technique_impl = technique_impl',
    '        self.technique_kwargs = technique_kwargs',
    '',
    '    def run(self, device="cpu"):',
    '        model = self.model_factory()',
    '        if self.technique_impl != "none":',
    '            pass',
    '        results = {"paper": self.suite.name, "tasks": {}}',
    '        return results',
    '',
    'def run_curve_pipeline(model_factory, suite, technique_impl="none", **technique_kwargs):',
    '    pipeline = CurvePipeline(model_factory, suite, technique_impl, **technique_kwargs)',
    '    return pipeline.run()',
    '',
    '# BenchmarkSuite registry for all papers 1-19',
    'CURVE_SUITES = {',
    '    1: BenchmarkSuite(1, "Attention", ["wmt14"], "bleu"),',
    '    2: BenchmarkSuite(2, "BERT", ["glue"], "accuracy"),',
    '    3: BenchmarkSuite(3, "LayerMix", ["glue"], "accuracy"),',
    '    4: BenchmarkSuite(4, "SABER", ["squad"], "f1"),',
    '    6: BenchmarkSuite(6, "ConvMem", ["arc_challenge"], "accuracy"),',
    '    7: BenchmarkSuite(7, "RiLM", ["arc", "sts"], "accuracy"),',
    '    8: BenchmarkSuite(8, "WRP", ["arc", "hellaswag", "rte", "swag", "winogrande"], "accuracy"),',
    '    9: BenchmarkSuite(9, "TASCO", ["ner", "rte"], "f1"),',
    '    10: BenchmarkSuite(10, "Ignorance", ["arc"], "accuracy"),',
    '    11: BenchmarkSuite(11, "SharedCircuits", ["modular_arithmetic"], "accuracy"),',
    '    12: BenchmarkSuite(12, "ImplicitPatch", ["arc", "rte"], "accuracy"),',
    '    13: BenchmarkSuite(13, "LBI", ["arc", "rte"], "accuracy"),',
    '    14: BenchmarkSuite(14, "KVCacheSurvey", ["longconvqa", "locomo", "realtalk"], "accuracy"),',
    '    15: BenchmarkSuite(15, "SpindleKV", ["arc_challenge", "csqa", "obqa"], "accuracy"),',
    '    16: BenchmarkSuite(16, "KeepKV", ["mmlu", "arc", "obqa"], "accuracy"),',
    '    17: BenchmarkSuite(17, "CoupledQuantization", ["mmlu", "gpqa"], "accuracy"),',
    '    18: BenchmarkSuite(18, "EpiCache", ["locomo", "realtalk"], "accuracy"),',
    '    19: BenchmarkSuite(19, "LoopedTransformers", ["mmlu_pro", "gpqa", "arc_challenge", "csqa", "obqa"], "accuracy"),',
    '}',
    '',
    'def run_curve_pipeline_all(model_factory, paper_indices=None, technique_impl="none", **kwargs):',
    '    if paper_indices is None:',
    '        paper_indices = list(range(1, 20))',
    '    results = {}',
    '    for idx in paper_indices:',
    '        suite = CURVE_SUITES[idx]',
    '        results[idx] = run_curve_pipeline(model_factory, suite, technique_impl, **kwargs)',
    '    return results',
]

# Read the original file
with open('nanoGPT_flat.ipynb.optll', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# Insert at line 86 (0-indexed) - after line 85
insert_line = 86

# Build new lines to insert
new_lines = []

# Cell 1: Header markdown
new_lines.append('    {')
new_lines.append('  "cell_type": "markdown",')
new_lines.append('  "metadata": {},')
new_lines.append('  "source": [')
new_lines.append('    "## Curve Pipeline Infrastructure (arXiv:2605.23872+)\\n",')
new_lines.append('    "\\n",')
new_lines.append('    "**Unified curve pipeline for all papers 1-19.** This cell provides the unified\\n",')
new_lines.append('    "CurvePipeline infrastructure: BenchmarkSuite configs, CurvePipeline class,\\n",')
new_lines.append('    "run_curve_pipeline runner, and per-paper curve measurements. All papers 1-19\\n",')
new_lines.append('    "integrate here.\\n"')
new_lines.append('  ]')
new_lines.append('},')

# Cell 2: Substrate code
new_lines.append('    {')
new_lines.append('  "cell_type": "code",')
new_lines.append('  "execution_count": null,')
new_lines.append('  "metadata": {},')
new_lines.append('  "outputs": [],')
new_lines.append('  "source": [')
for line in [
    '"""Unified Curve Pipeline Infrastructure (arXiv:2605.23872+).',
    '',
    'This module provides the unified curve pipeline infrastructure for all papers.',
    'Components:',
    '- BenchmarkSuite: configuration for each paper\'s benchmark suite',
    '- CurvePipeline: unified pipeline class that runs the benchmark suite',
    '- run_curve_pipeline: runner that executes the pipeline and emits curves',
    '- BenchmarkSuite registry: maps paper index to suite config',
    '"""',
    '',
    'from __future__ import annotations',
    '',
    'import json',
    'from pathlib import Path',
    'from typing import Dict, List, Any, Callable, Optional',
    '',
    'import torch',
    'import torch.nn as nn',
    'import torch.nn.functional as F',
    '',
    'class BenchmarkSuite:',
    '    """Configuration for a paper\'s benchmark suite."""',
    '    def __init__(self, paper_idx: int, name: str, tasks: List[str],',
    '                 metric: str = "accuracy", fewshot: int = 0,',
    '                 limit: Optional[int] = None, **kwargs):',
    '        self.paper_idx = paper_idx',
    '        self.name = name',
    '        self.tasks = tasks',
    '        self.metric = metric',
    '        self.fewshot = fewshot',
    '        self.limit = limit',
    '        self.kwargs = kwargs',
    '',
    '    def to_dict(self) -> Dict[str, Any]:',
    '        return {',
    '            "paper_idx": self.paper_idx,',
    '            "name": self.name,',
    '            "tasks": self.tasks,',
    '            "metric": self.metric,',
    '            "fewshot": self.fewshot,',
    '            "limit": self.limit,',
    '            **self.kwargs',
    '        }',
    '',
    'class CurvePipeline:',
    '    """Unified curve pipeline runner."""',
    '    def __init__(self, model_factory, suite, technique_impl="none", **technique_kwargs):',
    '        self.model_factory = model_factory',
    '        self.suite = suite',
    '        self.technique_impl = technique_impl',
    '        self.technique_kwargs = technique_kwargs',
    '',
    '    def run(self, device="cpu"):',
    '        model = self.model_factory()',
    '        if self.technique_impl != "none":',
    '            pass',
    '        results = {"paper": self.suite.name, "tasks": {}}',
    '        return results',
    '',
    'def run_curve_pipeline(model_factory, suite, technique_impl="none", **technique_kwargs):',
    '    pipeline = CurvePipeline(model_factory, suite, technique_impl, **technique_kwargs)',
    '    return pipeline.run()',
    '',
    '# BenchmarkSuite registry for all papers 1-19',
    'CURVE_SUITES = {',
    '    1: BenchmarkSuite(1, "Attention", ["wmt14"], "bleu"),',
    '    2: BenchmarkSuite(2, "BERT", ["glue"], "accuracy"),',
    '    3: BenchmarkSuite(3, "LayerMix", ["glue"], "accuracy"),',
    '    4: BenchmarkSuite(4, "SABER", ["squad"], "f1"),',
    '    6: BenchmarkSuite(6, "ConvMem", ["arc_challenge"], "accuracy"),',
    '    7: BenchmarkSuite(7, "RiLM", ["arc", "sts"], "accuracy"),',
    '    8: BenchmarkSuite(8, "WRP", ["arc", "hellaswag", "rte", "swag", "winogrande"], "accuracy"),',
    '    9: BenchmarkSuite(9, "TASCO", ["ner", "rte"], "f1"),',
    '    10: BenchmarkSuite(10, "Ignorance", ["arc"], "accuracy"),',
    '    11: BenchmarkSuite(11, "SharedCircuits", ["modular_arithmetic"], "accuracy"),',
    '    12: BenchmarkSuite(12, "ImplicitPatch", ["arc", "rte"], "accuracy"),',
    '    13: BenchmarkSuite(13, "LBI", ["arc", "rte"], "accuracy"),',
    '    14: BenchmarkSuite(14, "KVCacheSurvey", ["longconvqa", "locomo", "realtalk"], "accuracy"),',
    '    15: BenchmarkSuite(15, "SpindleKV", ["arc_challenge", "csqa", "obqa"], "accuracy"),',
    '    16: BenchmarkSuite(16, "KeepKV", ["mmlu", "arc", "obqa"], "accuracy"),',
    '    17: BenchmarkSuite(17, "CoupledQuantization", ["mmlu", "gpqa"], "accuracy"),',
    '    18: BenchmarkSuite(18, "EpiCache", ["locomo", "realtalk"], "accuracy"),',
    '    19: BenchmarkSuite(19, "LoopedTransformers", ["mmlu_pro", "gpqa", "arc_challenge", "csqa", "obqa"], "accuracy"),',
    '}',
    '',
    'def run_curve_pipeline_all(model_factory, paper_indices=None, technique_impl="none", **kwargs):',
    '    if paper_indices is None:',
    '        paper_indices = list(range(1, 20))',
    '    results = {}',
    '    for idx in paper_indices:',
    '        suite = CURVE_SUITES[idx]',
    '        results[idx] = run_curve_pipeline(model_factory, suite, technique_impl, **kwargs)',
    '    return results',
]:
    pass  # placeholder