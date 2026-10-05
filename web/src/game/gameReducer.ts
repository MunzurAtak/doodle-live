/**
 * Pure game state machine (SPEC §7.4). Time is passed in with every action (`now`, in
 * ms), so the reducer has no side effects and is easy to test.
 *
 *   idle -> countdown (3 s) -> drawing -> roundWon | roundLost -> countdown ... -> gameOver
 */
import {
  COUNTDOWN_SECONDS,
  ROUNDS_PER_GAME,
  ROUND_SECONDS,
  WIN_PROBABILITY,
  scoreRound,
} from './scoring'

export type Phase = 'idle' | 'countdown' | 'drawing' | 'roundWon' | 'roundLost' | 'gameOver'

/** The model's output after a finished stroke (one point per stroke on the chart). */
export interface StrokeSnapshot {
  readonly strokes: number
  readonly probs: readonly number[]
}

export type RoundOutcome = 'guessed' | 'timeout' | 'skipped'

export interface RoundResult {
  readonly word: string
  readonly won: boolean
  readonly outcome: RoundOutcome
  readonly score: number
  readonly strokes: number
  readonly secondsLeft: number
  readonly history: readonly StrokeSnapshot[]
}

export interface GameState {
  readonly phase: Phase
  readonly words: readonly string[]
  readonly round: number
  /** countdown: when drawing starts. drawing: when time runs out. */
  readonly deadline: number
  readonly strokes: number
  readonly history: readonly StrokeSnapshot[]
  readonly results: readonly RoundResult[]
  readonly score: number
}

export type GameAction =
  | { readonly type: 'start'; readonly words: readonly string[]; readonly now: number }
  | { readonly type: 'tick'; readonly now: number }
  | {
      readonly type: 'prediction'
      readonly now: number
      readonly label: string | null
      readonly prob: number
      readonly probs: readonly number[]
      readonly strokes: number
      /** true when no stroke is in progress (pen-up, undo, clear). */
      readonly final: boolean
    }
  | { readonly type: 'skip'; readonly now: number }
  | { readonly type: 'next'; readonly now: number }
  | { readonly type: 'quit' }

export const INITIAL_GAME: GameState = {
  phase: 'idle',
  words: [],
  round: 0,
  deadline: 0,
  strokes: 0,
  history: [],
  results: [],
  score: 0,
}

export const currentWord = (state: GameState): string | null => state.words[state.round] ?? null

export const secondsLeft = (state: GameState, now: number): number =>
  Math.max(0, (state.deadline - now) / 1000)

const startCountdown = (state: GameState, round: number, now: number): GameState => ({
  ...state,
  phase: 'countdown',
  round,
  deadline: now + COUNTDOWN_SECONDS * 1000,
  strokes: 0,
  history: [],
})

function endRound(state: GameState, outcome: RoundOutcome, now: number): GameState {
  const won = outcome === 'guessed'
  const left = won ? secondsLeft(state, now) : 0
  const score = scoreRound(won, left, state.strokes)
  const result: RoundResult = {
    word: currentWord(state) ?? '',
    won,
    outcome,
    score,
    strokes: state.strokes,
    secondsLeft: Math.floor(left),
    history: state.history,
  }
  return {
    ...state,
    phase: won ? 'roundWon' : 'roundLost',
    results: [...state.results, result],
    score: state.score + score,
  }
}

function recordSnapshot(history: readonly StrokeSnapshot[], snapshot: StrokeSnapshot) {
  // One snapshot per stroke count; an undo truncates the history.
  return [...history.filter((s) => s.strokes < snapshot.strokes), snapshot]
}

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'start': {
      if (action.words.length < ROUNDS_PER_GAME) throw new Error('not enough words')
      const fresh = { ...INITIAL_GAME, words: action.words.slice(0, ROUNDS_PER_GAME) }
      return startCountdown(fresh, 0, action.now)
    }
    case 'tick':
      if (state.phase === 'countdown' && action.now >= state.deadline) {
        return { ...state, phase: 'drawing', deadline: action.now + ROUND_SECONDS * 1000 }
      }
      if (state.phase === 'drawing' && action.now >= state.deadline) {
        return endRound(state, 'timeout', action.now)
      }
      return state
    case 'prediction': {
      // Free drawing (idle) records history for the chart but has no target to win.
      if (state.phase !== 'drawing' && state.phase !== 'idle') return state
      const hit =
        state.phase === 'drawing' &&
        action.strokes > 0 &&
        action.label === currentWord(state) &&
        action.prob >= WIN_PROBABILITY
      let next: GameState = { ...state, strokes: action.strokes }
      // Record finished strokes, and the winning moment even if it came mid-stroke.
      if (action.final || hit) {
        next =
          action.strokes === 0
            ? { ...next, history: [] }
            : {
                ...next,
                history: recordSnapshot(state.history, {
                  strokes: action.strokes,
                  probs: action.probs,
                }),
              }
      }
      return hit ? endRound(next, 'guessed', action.now) : next
    }
    case 'skip':
      return state.phase === 'drawing' ? endRound(state, 'skipped', action.now) : state
    case 'next':
      if (state.phase !== 'roundWon' && state.phase !== 'roundLost') return state
      return state.round + 1 >= state.words.length
        ? { ...state, phase: 'gameOver' }
        : startCountdown(state, state.round + 1, action.now)
    case 'quit':
      return INITIAL_GAME
  }
}
