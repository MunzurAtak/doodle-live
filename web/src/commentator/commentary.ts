/**
 * Scripted commentator state (pure). Turns game phases, predictions and clock ticks into
 * events (SPEC §8.2) and keeps the line to show for the most important one. Randomness
 * is passed in with the action so the reducer stays deterministic.
 *
 * Phase 6 swaps the line source for the LLM orchestrator; the event logic stays.
 */
import type { Phase } from '../game/gameReducer'
import type { Guess } from '../ml/inference'
import {
  EMPTY_MEMORY,
  type EventMemory,
  type GameEvent,
  detectPredictionEvents,
  detectStuck,
  makeEvent,
} from './events'
import { scriptedLine } from './fallbackLines'

export type Mood = 'idle' | 'thinking' | 'sure' | 'happy' | 'sad'

export interface CommentaryState {
  readonly memory: EventMemory
  readonly lastStrokeAt: number
  readonly strokes: number
  readonly line: string | null
  readonly mood: Mood
  readonly event: GameEvent | null
}

export type CommentaryAction =
  | {
      readonly type: 'phase'
      readonly phase: Phase
      readonly word: string | null
      readonly label: string | undefined
      readonly now: number
      readonly rand: number
    }
  | { readonly type: 'prediction'; readonly top: readonly Guess[]; readonly rand: number }
  | { readonly type: 'strokes'; readonly strokes: number; readonly now: number }
  | {
      readonly type: 'tick'
      readonly now: number
      readonly top: readonly Guess[]
      readonly rand: number
    }

export const INITIAL_COMMENTARY: CommentaryState = {
  memory: EMPTY_MEMORY,
  lastStrokeAt: 0,
  strokes: 0,
  line: null,
  mood: 'idle',
  event: null,
}

const MOOD: Record<GameEvent['type'], Mood> = {
  ROUND_START: 'idle',
  FIRST_GUESS: 'thinking',
  GUESS_CHANGED: 'thinking',
  STUCK: 'thinking',
  CONFIDENT: 'sure',
  ROUND_WON: 'happy',
  ROUND_LOST: 'sad',
}

function say(state: CommentaryState, events: readonly GameEvent[], rand: number): CommentaryState {
  if (events.length === 0) return state
  const event = events.reduce((a, b) => (b.priority >= a.priority ? b : a))
  const line = scriptedLine(event, state.line, () => rand)
  return { ...state, line, mood: MOOD[event.type], event }
}

export function commentaryReducer(
  state: CommentaryState,
  action: CommentaryAction,
): CommentaryState {
  switch (action.type) {
    case 'phase':
      switch (action.phase) {
        case 'drawing':
          return say(
            { ...state, memory: EMPTY_MEMORY, lastStrokeAt: action.now, strokes: 0 },
            [makeEvent('ROUND_START')],
            action.rand,
          )
        case 'roundWon':
        case 'roundLost':
          return say(
            state,
            [
              makeEvent(action.phase === 'roundWon' ? 'ROUND_WON' : 'ROUND_LOST', {
                label: action.label,
                answer: action.word ?? undefined,
              }),
            ],
            action.rand,
          )
        case 'idle':
        case 'countdown':
          return { ...INITIAL_COMMENTARY, lastStrokeAt: action.now }
        default:
          return state
      }
    case 'prediction': {
      const out = detectPredictionEvents(state.memory, action.top)
      return say({ ...state, memory: out.memory }, out.events, action.rand)
    }
    case 'strokes':
      if (action.strokes === state.strokes) return state
      return {
        ...state,
        strokes: action.strokes,
        lastStrokeAt: action.now,
        // An empty canvas starts a fresh conversation about the next drawing.
        memory: action.strokes === 0 ? EMPTY_MEMORY : state.memory,
      }
    case 'tick': {
      if (state.strokes === 0) return state
      const out = detectStuck(state.memory, {
        now: action.now,
        lastStrokeAt: state.lastStrokeAt,
        strokes: state.strokes,
        topProb: action.top[0]?.prob ?? 0,
        label: action.top[0]?.label,
      })
      return say({ ...state, memory: out.memory }, out.events, action.rand)
    }
  }
}
