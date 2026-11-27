# Agent memory guard

Host-only DSH plugin. It registers one system-prompt section whose text is read
from `AGENTS_MEMORY.md` at every prompt assembly, so the memory is in context
before every model step, not just at session start.

The memory path is configured in `cordis.patch.yml`.
