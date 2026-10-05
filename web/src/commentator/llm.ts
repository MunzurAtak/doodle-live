/** The orchestrator talks to any model through this interface (SPEC §8.3). */
import { type GameEvent, type GameEventType, makeEvent } from './events'
import { scriptedLine } from './fallbackLines'
import { type ChatMessage, GENERATION } from './prompt'

export interface LLMClient {
  /** Stream text chunks. Must stop promptly when `signal` aborts. */
  stream(messages: readonly ChatMessage[], signal: AbortSignal): AsyncIterable<string>
}

/** The part of WebLLM's engine we use (kept minimal so tests can fake it). */
export interface ChatEngine {
  chat: {
    completions: {
      create(request: {
        messages: ChatMessage[]
        stream: true
        temperature: number
        max_tokens: number
      }): Promise<AsyncIterable<{ choices: Array<{ delta?: { content?: string | null } }> }>>
    }
  }
  interruptGenerate(): void | Promise<void>
}

export class WebLLMClient implements LLMClient {
  private readonly engine: ChatEngine

  constructor(engine: ChatEngine) {
    this.engine = engine
  }

  async *stream(messages: readonly ChatMessage[], signal: AbortSignal): AsyncIterable<string> {
    if (signal.aborted) return
    const chunks = await this.engine.chat.completions.create({
      messages: [...messages],
      stream: true,
      ...GENERATION,
    })
    const onAbort = () => void this.engine.interruptGenerate()
    signal.addEventListener('abort', onAbort, { once: true })
    try {
      for await (const chunk of chunks) {
        if (signal.aborted) return
        const text = chunk.choices[0]?.delta?.content
        if (text) yield text
      }
    } finally {
      signal.removeEventListener('abort', onAbort)
    }
  }
}

/** Reads the event back out of the prompt's JSON state. */
export function eventFromMessages(messages: readonly ChatMessage[]): {
  event: GameEvent
  previous: string | null
} {
  const user = [...messages].reverse().find((m) => m.role === 'user')
  const state = JSON.parse(user?.content ?? '{}') as {
    event?: GameEventType
    candidates?: Array<{ label: string }>
    answer?: string
    previous_lines?: string[]
  }
  const event = makeEvent(state.event ?? 'ROUND_START', {
    label: state.candidates?.[0]?.label,
    answer: state.answer,
  })
  return { event, previous: state.previous_lines?.at(-1) ?? null }
}

/** Scripted lines behind the same interface: used without WebGPU or before the model
 * has loaded. */
export class ScriptedClient implements LLMClient {
  private readonly random: () => number

  constructor(random: () => number = Math.random) {
    this.random = random
  }

  async *stream(messages: readonly ChatMessage[], signal: AbortSignal): AsyncIterable<string> {
    if (signal.aborted) return
    const { event, previous } = eventFromMessages(messages)
    yield scriptedLine(event, previous, this.random)
  }
}
