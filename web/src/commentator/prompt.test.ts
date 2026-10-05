import { describe, expect, it } from 'vitest'
import { makeEvent } from './events'
import { getPersonality } from './personalities'
import { buildMessages, buildUserState, confidenceWord, trimToSentence } from './prompt'

const ctx = {
  strokes: 4,
  secondsLeft: 10.2,
  candidates: [
    { label: 'windmill', prob: 0.72 },
    { label: 'flower', prob: 0.31 },
    { label: 'star', prob: 0.05 },
    { label: 'sun', prob: 0.01 },
  ],
  previousLines: ['a', 'b', 'c', 'Is that a flower? Lots of petals...'],
  answer: 'windmill',
}

describe('prompt', () => {
  it('maps probabilities to words', () => {
    expect([0.6, 0.59, 0.3, 0.29].map(confidenceWord)).toEqual(['high', 'medium', 'medium', 'low'])
  })

  it('builds the compact state from the spec', () => {
    const state = buildUserState(makeEvent('GUESS_CHANGED', { label: 'windmill' }), ctx)
    expect(state).toMatchObject({
      event: 'GUESS_CHANGED',
      strokes: 4,
      seconds_left: 11,
      candidates: [
        { label: 'windmill', confidence: 'high' },
        { label: 'flower', confidence: 'medium' },
        { label: 'star', confidence: 'low' },
      ],
      previous_lines: ['b', 'c', 'Is that a flower? Lots of petals...'],
    })
  })

  it('never reveals the answer while the user is drawing', () => {
    for (const type of [
      'ROUND_START',
      'FIRST_GUESS',
      'GUESS_CHANGED',
      'CONFIDENT',
      'STUCK',
    ] as const) {
      const state = buildUserState(makeEvent(type), { ...ctx, candidates: [] })
      expect(state).not.toHaveProperty('answer')
      expect(JSON.stringify(state)).not.toContain('windmill')
    }
  })

  it('reveals the answer when the round ends', () => {
    expect(buildUserState(makeEvent('ROUND_WON'), ctx).answer).toBe('windmill')
    expect(buildUserState(makeEvent('ROUND_LOST'), ctx).answer).toBe('windmill')
  })

  it('puts rules + personality in the system message and JSON in the user message', () => {
    const messages = buildMessages(makeEvent('FIRST_GUESS'), ctx, getPersonality('critic'))
    expect(messages.map((m) => m.role)).toEqual(['system', 'user'])
    expect(messages[0].content).toContain('Never invent other objects')
    expect(messages[0].content).toContain('art critic')
    expect(JSON.parse(messages[1].content).event).toBe('FIRST_GUESS')
  })

  it('trims to the last complete sentence', () => {
    expect(trimToSentence('A windmill! I was stuck on flo')).toBe('A windmill!')
    expect(trimToSentence('Is that a "cat?" Hmm')).toBe('Is that a "cat?"')
    expect(trimToSentence('  no end at all ')).toBe('no end at all')
    expect(trimToSentence('Two.  Sentences.')).toBe('Two. Sentences.')
  })
})
