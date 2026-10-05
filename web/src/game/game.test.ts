import { describe, expect, it } from 'vitest'
import {
  type GameAction,
  type GameState,
  INITIAL_GAME,
  currentWord,
  gameReducer,
  secondsLeft,
} from './gameReducer'
import { scoreRound } from './scoring'
import { pickWords } from './words'

const WORDS = ['cat', 'sun', 'tree', 'key', 'cup', 'hat']
const run = (state: GameState, ...actions: GameAction[]) => actions.reduce(gameReducer, state)
const predict = (now: number, label: string, prob: number, strokes: number, final = true) =>
  ({ type: 'prediction', now, label, prob, probs: [prob], strokes, final }) as const

/** A game that is drawing round 0, started at t=0 (drawing from t=3000 until t=23000). */
const drawing = () =>
  run(INITIAL_GAME, { type: 'start', words: WORDS, now: 0 }, { type: 'tick', now: 3000 })

describe('pickWords', () => {
  it('returns n distinct words from the categories', () => {
    const cats = Array.from({ length: 50 }, (_, i) => `w${i}`)
    const words = pickWords(cats, 6)
    expect(new Set(words).size).toBe(6)
    expect(words.every((w) => cats.includes(w))).toBe(true)
  })

  it('is deterministic for a given random source and errors when too few', () => {
    const fixed = () => 0
    expect(pickWords(['a', 'b', 'c'], 2, fixed)).toEqual(['a', 'b'])
    expect(() => pickWords(['a'], 2)).toThrow()
  })
})

describe('scoreRound', () => {
  it('follows the spec', () => {
    expect(scoreRound(true, 12.9, 5)).toBe(160)
    expect(scoreRound(true, 12.9, 3)).toBe(210)
    expect(scoreRound(true, 0, 1)).toBe(150)
    expect(scoreRound(false, 15, 1)).toBe(0)
  })
})

describe('gameReducer', () => {
  it('counts down 3 s, then gives 20 s to draw', () => {
    const counting = run(INITIAL_GAME, { type: 'start', words: WORDS, now: 0 })
    expect(counting.phase).toBe('countdown')
    expect(run(counting, { type: 'tick', now: 2999 }).phase).toBe('countdown')
    const state = drawing()
    expect(state.phase).toBe('drawing')
    expect(currentWord(state)).toBe('cat')
    expect(secondsLeft(state, 13000)).toBe(10)
  })

  it('wins when the target is top-1 with p >= 0.5 and scores the round', () => {
    const state = run(drawing(), predict(4000, 'cat', 0.49, 1), predict(6000, 'cat', 0.8, 2))
    expect(state.phase).toBe('roundWon')
    expect(state.results[0]).toMatchObject({
      word: 'cat',
      won: true,
      outcome: 'guessed',
      strokes: 2,
      secondsLeft: 17,
    })
    expect(state.score).toBe(100 + 5 * 17 + 50)
  })

  it('records the winning prediction on the chart even mid-stroke', () => {
    const state = run(drawing(), predict(4000, 'cat', 0.2, 1), predict(5000, 'cat', 0.7, 2, false))
    expect(state.phase).toBe('roundWon')
    expect(state.results[0].history.map((s) => s.strokes)).toEqual([1, 2])
  })

  it('records history but never wins in free drawing (idle)', () => {
    const state = run(INITIAL_GAME, predict(0, 'cat', 0.99, 1))
    expect(state.phase).toBe('idle')
    expect(state.history).toHaveLength(1)
  })

  it('does not win on another label or before drawing starts', () => {
    expect(run(drawing(), predict(4000, 'dog', 0.99, 1)).phase).toBe('drawing')
    const counting = run(INITIAL_GAME, { type: 'start', words: WORDS, now: 0 })
    expect(run(counting, predict(1000, 'cat', 0.99, 1)).phase).toBe('countdown')
  })

  it('loses when time runs out or the round is skipped', () => {
    const timedOut = run(drawing(), { type: 'tick', now: 23000 })
    expect(timedOut.phase).toBe('roundLost')
    expect(timedOut.results[0]).toMatchObject({ won: false, outcome: 'timeout', score: 0 })
    const skipped = run(drawing(), { type: 'skip', now: 5000 })
    expect(skipped.phase).toBe('roundLost')
    expect(skipped.results[0].outcome).toBe('skipped')
  })

  it('keeps one snapshot per stroke and truncates on undo', () => {
    const state = run(
      drawing(),
      predict(4000, 'dog', 0.1, 1),
      predict(4100, 'dog', 0.2, 2, false), // in progress: not recorded
      predict(5000, 'dog', 0.3, 2),
      predict(6000, 'dog', 0.4, 3),
      predict(7000, 'dog', 0.35, 2), // undo
    )
    expect(state.history.map((s) => [s.strokes, s.probs[0]])).toEqual([
      [1, 0.1],
      [2, 0.35],
    ])
    expect(run(state, predict(8000, 'dog', 0, 0)).history).toEqual([])
  })

  it('plays 6 rounds and ends the game', () => {
    let state = drawing()
    for (let round = 0; round < 6; round++) {
      const t = 100_000 * (round + 1)
      state = run(state, { type: 'skip', now: t }, { type: 'next', now: t })
      if (round < 5) {
        expect(state.phase).toBe('countdown')
        expect(state.round).toBe(round + 1)
        expect(state.strokes).toBe(0)
        state = run(state, { type: 'tick', now: t + 3000 })
      }
    }
    expect(state.phase).toBe('gameOver')
    expect(state.results).toHaveLength(6)
    expect(run(state, { type: 'quit' })).toBe(INITIAL_GAME)
  })
})
