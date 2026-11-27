import fs from 'node:fs'

export const name = 'memory-guard'
export const inject = ['systemPrompt']

/**
 * Adds one system-prompt section whose text is re-read from the memory file at
 * every assembly. The model therefore sees the current memory before every
 * step/query, not only at session start.
 */
export function apply(ctx, config) {
  const file = config?.path || 'C:/Users/AMBE/Desktop/Engineering/nanoGPT/AGENTS_MEMORY.md'
  const read = () => {
    try {
      return fs.readFileSync(file, 'utf8')
    } catch (error) {
      return 'MANDATORY AGENT MEMORY IS MISSING. Read ' + file + ' and fail closed. ' +
        String(error?.message || error)
    }
  }
  ctx.effect(() => ctx.systemPrompt.section({
    name: 'local:agent-memory',
    order: 9000,
    interpolate: false,
    text: () => 'MANDATORY AGENT MEMORY (re-read from ' + file + ' for this step):\n\n' + read(),
  }))
}
