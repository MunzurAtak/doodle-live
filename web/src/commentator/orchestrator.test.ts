import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { type GameEventType, makeEvent } from './events'
import { eventFromMessages, type LLMClient } from './llm'
import { type CommentMetric, type CommentView, CommentaryOrchestrator } from './orchestrator'
import { type ChatMessage } from './prompt'

/** Emits "<EVENT> w1 w2 ... ." one token every `tokenMs`. */
class FakeLLM implements LLMClient {
  tokenMs: number
  words: number
  calls = 0
  fail = false
  constructor(tokenMs = 100, words = 5) {
    this.tokenMs = tokenMs
    this.words = words
  }
  async *stream(messages: readonly ChatMessage[], signal: AbortSignal) {
    this.calls++
    if (this.fail) throw new Error('model crashed')
    const { event } = eventFromMessages(messages)
    const tokens = [event.type, ...Array.from({ length: this.words }, (_, i) => ` w${i}`), '.']
    for (const token of tokens) {
      await new Promise((r) => setTimeout(r, this.tokenMs))
      if (signal.aborted) return
      yield token
    }
  }
}

function setup(llm = new FakeLLM()) {
  const views: CommentView[] = []
  const metrics: CommentMetric[] = []
  const o = new CommentaryOrchestrator({
    client: llm,
    source: 'llm',
    buildMessages: (q) => [
      { role: 'user', content: JSON.stringify({ event: q.event.type, candidates: [] }) },
    ],
    fallbackLine: (e) => `fallback ${e.type}`,
    onView: (v) => views.push(v),
    onMetric: (m) => metrics.push(m),
    now: () => Date.now(),
  })
  const push = (type: GameEventType, round = 0) => o.push(makeEvent(type), round)
  /** Events of the comments that were started, in order. */
  const started = () => [...new Set(views.map((v) => `${v.id}:${v.event.type}`))]
  const finalTexts = () => views.filter((v) => !v.streaming).map((v) => v.text)
  return { o, llm, views, metrics, push, started, finalTexts }
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(0)
})
afterEach(() => vi.useRealTimers())

describe('CommentaryOrchestrator', () => {
  it('streams a comment and finishes with the trimmed text and a metric', async () => {
    const { push, views, metrics, finalTexts } = setup()
    push('FIRST_GUESS')
    await vi.advanceTimersByTimeAsync(150)
    expect(views.at(-1)).toMatchObject({ text: 'FIRST_GUESS', streaming: true })
    await vi.advanceTimersByTimeAsync(1000)
    expect(finalTexts()).toEqual(['FIRST_GUESS w0 w1 w2 w3 w4.'])
    expect(metrics[0]).toMatchObject({ event: 'FIRST_GUESS', interrupted: false, fallback: false })
    expect(metrics[0].firstTokenMs).toBe(100)
  })

  it('respects the 1.5 s cooldown between comment starts', async () => {
    const { push, started } = setup(new FakeLLM(10, 1))
    push('FIRST_GUESS')
    await vi.advanceTimersByTimeAsync(200) // first comment done
    push('GUESS_CHANGED')
    await vi.advanceTimersByTimeAsync(1000)
    expect(started()).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(400)
    expect(started()).toEqual(['1:FIRST_GUESS', '2:GUESS_CHANGED'])
  })

  it('keeps only the latest pending event', async () => {
    const { push, started } = setup(new FakeLLM(10, 1))
    push('FIRST_GUESS')
    await vi.advanceTimersByTimeAsync(100)
    push('GUESS_CHANGED')
    push('STUCK')
    await vi.advanceTimersByTimeAsync(3000)
    expect(started()).toEqual(['1:FIRST_GUESS', '2:STUCK'])
  })

  it('interrupts an equal-priority line only after it was visible for 0.8 s', async () => {
    const { push, started, metrics } = setup(new FakeLLM(200, 20))
    push('FIRST_GUESS') // first token at 200 ms
    await vi.advanceTimersByTimeAsync(1600) // cooldown passed, visible 1.4 s
    push('GUESS_CHANGED')
    await vi.advanceTimersByTimeAsync(50)
    expect(started()).toEqual(['1:FIRST_GUESS', '2:GUESS_CHANGED'])
    await vi.advanceTimersByTimeAsync(250) // the aborted stream stops at its next token
    expect(metrics[0]).toMatchObject({ event: 'FIRST_GUESS', interrupted: true })
  })

  it('does not let a lower-priority event interrupt; it waits until the line ends', async () => {
    const { push, started } = setup(new FakeLLM(200, 10))
    push('CONFIDENT') // priority 2, runs ~2.4 s
    await vi.advanceTimersByTimeAsync(1600)
    push('GUESS_CHANGED') // priority 1
    await vi.advanceTimersByTimeAsync(500)
    expect(started()).toEqual(['1:CONFIDENT'])
    await vi.advanceTimersByTimeAsync(1000)
    expect(started()).toEqual(['1:CONFIDENT', '2:GUESS_CHANGED'])
  })

  it('priority-3 events interrupt immediately, ignore the cooldown and clear pending', async () => {
    const { push, started, metrics } = setup(new FakeLLM(200, 20))
    push('FIRST_GUESS')
    await vi.advanceTimersByTimeAsync(300)
    push('GUESS_CHANGED') // pending (cooldown + visibility)
    push('ROUND_WON')
    await vi.advanceTimersByTimeAsync(10)
    expect(started()).toEqual(['1:FIRST_GUESS', '2:ROUND_WON'])
    await vi.advanceTimersByTimeAsync(10_000)
    expect(started()).toHaveLength(2) // GUESS_CHANGED was dropped
    expect(metrics[0].interrupted).toBe(true)
  })

  it('drops lines from previous rounds', async () => {
    const { o, push, started, views } = setup(new FakeLLM(200, 20))
    push('FIRST_GUESS', 0)
    await vi.advanceTimersByTimeAsync(300)
    const before = views.length
    o.setRound(1)
    push('GUESS_CHANGED', 0) // stale round
    await vi.advanceTimersByTimeAsync(5000)
    expect(views.length).toBe(before)
    push('ROUND_START', 1)
    await vi.advanceTimersByTimeAsync(5000)
    expect(started().at(-1)).toBe('2:ROUND_START')
  })

  it('falls back to a scripted line when the model fails', async () => {
    const llm = new FakeLLM()
    llm.fail = true
    const { push, finalTexts, metrics } = setup(llm)
    push('STUCK')
    await vi.advanceTimersByTimeAsync(10)
    expect(finalTexts()).toEqual(['fallback STUCK'])
    expect(metrics[0].fallback).toBe(true)
  })
})
