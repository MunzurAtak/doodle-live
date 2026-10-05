import { describe, expect, it } from 'vitest'
import {
  EMPTY_MEMORY,
  type EventMemory,
  detectPredictionEvents,
  detectStuck,
  makeEvent,
} from './events'
import { FALLBACK_LINES, fillTemplate, scriptedLine } from './fallbackLines'

const top = (label: string, prob: number) => [{ label, prob }]

function feed(steps: Array<[string, number]>) {
  let memory: EventMemory = EMPTY_MEMORY
  const types: string[] = []
  for (const [label, prob] of steps) {
    const out = detectPredictionEvents(memory, top(label, prob))
    memory = out.memory
    types.push(out.events.map((e) => `${e.type}:${e.label}`).join('+') || '-')
  }
  return types
}

describe('detectPredictionEvents', () => {
  it.each([
    ['nothing below the first-guess threshold', [['cat', 0.1]], ['-']],
    [
      'first guess once',
      [
        ['cat', 0.2],
        ['cat', 0.3],
      ],
      ['FIRST_GUESS:cat', '-'],
    ],
    [
      'guess changes need p >= 0.25',
      [
        ['cat', 0.2],
        ['dog', 0.2],
        ['dog', 0.3],
        ['cat', 0.3],
      ],
      ['FIRST_GUESS:cat', '-', 'GUESS_CHANGED:dog', 'GUESS_CHANGED:cat'],
    ],
    [
      'confident fires once per label',
      [
        ['cat', 0.2],
        ['cat', 0.75],
        ['cat', 0.9],
        ['dog', 0.8],
        ['cat', 0.8],
      ],
      [
        'FIRST_GUESS:cat',
        'CONFIDENT:cat',
        '-',
        'GUESS_CHANGED:dog+CONFIDENT:dog',
        'GUESS_CHANGED:cat',
      ],
    ],
    ['first guess and confident together', [['sun', 0.9]], ['FIRST_GUESS:sun+CONFIDENT:sun']],
  ] as const)('%s', (_name, steps, expected) => {
    expect(feed(steps as unknown as Array<[string, number]>)).toEqual(expected)
  })

  it('ignores empty predictions', () => {
    expect(detectPredictionEvents(EMPTY_MEMORY, []).events).toEqual([])
  })

  it('assigns priorities from the spec', () => {
    expect(makeEvent('ROUND_WON').priority).toBe(3)
    expect(makeEvent('CONFIDENT').priority).toBe(2)
    expect(makeEvent('GUESS_CHANGED').priority).toBe(1)
  })
})

describe('detectStuck', () => {
  const ctx = { now: 10_000, lastStrokeAt: 5_000, strokes: 2, topProb: 0.3, label: 'cat' }

  it('fires after 4 s without a stroke while unsure, once per pause', () => {
    const first = detectStuck(EMPTY_MEMORY, ctx)
    expect(first.events.map((e) => e.type)).toEqual(['STUCK'])
    expect(detectStuck(first.memory, { ...ctx, now: 20_000 }).events).toEqual([])
    expect(detectStuck(first.memory, { ...ctx, strokes: 3, now: 20_000 }).events).toHaveLength(1)
  })

  it('does not fire too early or when the model is confident', () => {
    expect(detectStuck(EMPTY_MEMORY, { ...ctx, now: 8_000 }).events).toEqual([])
    expect(detectStuck(EMPTY_MEMORY, { ...ctx, topProb: 0.6 }).events).toEqual([])
  })
})

describe('scripted lines', () => {
  it('has lines for every event type', () => {
    for (const lines of Object.values(FALLBACK_LINES))
      expect(lines.length).toBeGreaterThanOrEqual(5)
  })

  it('fills placeholders and capitalises', () => {
    expect(
      fillTemplate('{answer}! Great drawing.', makeEvent('ROUND_WON', { answer: 'windmill' })),
    ).toBe('Windmill! Great drawing.')
  })

  it('avoids repeating the previous line', () => {
    const event = makeEvent('FIRST_GUESS', { label: 'cat' })
    const first = scriptedLine(event, null, () => 0)
    expect(scriptedLine(event, first, () => 0)).not.toBe(first)
  })
})

describe('commentaryReducer', () => {
  it('emits round start, guesses, stuck and the result as numbered events', async () => {
    const { INITIAL_COMMENTARY, commentaryReducer } = await import('./commentary')
    let s = commentaryReducer(INITIAL_COMMENTARY, {
      type: 'phase',
      phase: 'drawing',
      word: 'cat',
      label: undefined,
      now: 0,
    })
    expect([s.event?.type, s.seq]).toEqual(['ROUND_START', 1])
    s = commentaryReducer(s, { type: 'strokes', strokes: 1, now: 500 })
    s = commentaryReducer(s, { type: 'prediction', top: top('dog', 0.3) })
    expect([s.event?.type, s.event?.label, s.seq]).toEqual(['FIRST_GUESS', 'dog', 2])
    s = commentaryReducer(s, { type: 'tick', now: 5000, top: top('dog', 0.3) })
    expect(s.event?.type).toBe('STUCK')
    s = commentaryReducer(s, {
      type: 'phase',
      phase: 'roundWon',
      word: 'cat',
      label: 'cat',
      now: 6000,
    })
    expect(s.event).toMatchObject({ type: 'ROUND_WON', answer: 'cat', priority: 3 })
  })

  it('starts a new epoch on countdown / free drawing and resets on an empty canvas', async () => {
    const { INITIAL_COMMENTARY, commentaryReducer } = await import('./commentary')
    let s = commentaryReducer(INITIAL_COMMENTARY, { type: 'strokes', strokes: 2, now: 0 })
    s = commentaryReducer(s, { type: 'prediction', top: top('sun', 0.4) })
    s = commentaryReducer(s, { type: 'strokes', strokes: 0, now: 10 })
    expect(s.memory.firstGuessMade).toBe(false)
    const next = commentaryReducer(s, {
      type: 'phase',
      phase: 'countdown',
      word: 'cat',
      label: undefined,
      now: 20,
    })
    expect(next.epoch).toBe(s.epoch + 1)
    expect(next.seq).toBe(s.seq)
  })
})
