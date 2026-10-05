/**
 * Commentary event state (pure). Turns game phases, predictions and clock ticks into
 * events (SPEC §8.2). The orchestrator decides when and how to voice them.
 *
 * `seq` increases with every emitted event; `epoch` increases whenever a new drawing
 * session starts (each round's countdown, or going back to free drawing), so lines from
 * an older session can be dropped.
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

export type Mood = 'idle' | 'thinking' | 'sure' | 'happy' | 'sad'

export interface CommentaryState {
  readonly memory: EventMemory
  readonly lastStrokeAt: number
  readonly strokes: number
  readonly event: GameEvent | null
  readonly seq: number
  readonly epoch: number
}

export type CommentaryAction =
  | {
      readonly type: 'phase'
      readonly phase: Phase
      readonly word: string | null
      readonly label: string | undefined
      readonly now: number
    }
  | { readonly type: 'prediction'; readonly top: readonly Guess[] }
  | { readonly type: 'strokes'; readonly strokes: number; readonly now: number }
  | { readonly type: 'tick'; readonly now: number; readonly top: readonly Guess[] }

export const INITIAL_COMMENTARY: CommentaryState = {
  memory: EMPTY_MEMORY,
  lastStrokeAt: 0,
  strokes: 0,
  event: null,
  seq: 0,
  epoch: 0,
}

export const MOOD: Record<GameEvent['type'], Mood> = {
  ROUND_START: 'idle',
  FIRST_GUESS: 'thinking',
  GUESS_CHANGED: 'thinking',
  STUCK: 'thinking',
  CONFIDENT: 'sure',
  ROUND_WON: 'happy',
  ROUND_LOST: 'sad',
}

/** Emit the most important of several simultaneous events. */
function emit(state: CommentaryState, events: readonly GameEvent[]): CommentaryState {
  if (events.length === 0) return state
  const event = events.reduce((a, b) => (b.priority >= a.priority ? b : a))
  return { ...state, event, seq: state.seq + 1 }
}

export function commentaryReducer(
  state: CommentaryState,
  action: CommentaryAction,
): CommentaryState {
  switch (action.type) {
    case 'phase':
      switch (action.phase) {
        case 'drawing':
          return emit({ ...state, memory: EMPTY_MEMORY, lastStrokeAt: action.now, strokes: 0 }, [
            makeEvent('ROUND_START'),
          ])
        case 'roundWon':
        case 'roundLost':
          return emit(state, [
            makeEvent(action.phase === 'roundWon' ? 'ROUND_WON' : 'ROUND_LOST', {
              label: action.label,
              answer: action.word ?? undefined,
            }),
          ])
        case 'idle':
        case 'countdown':
          return {
            ...INITIAL_COMMENTARY,
            seq: state.seq,
            epoch: state.epoch + 1,
            lastStrokeAt: action.now,
          }
        default:
          return state
      }
    case 'prediction': {
      const out = detectPredictionEvents(state.memory, action.top)
      return emit({ ...state, memory: out.memory }, out.events)
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
      return emit({ ...state, memory: out.memory }, out.events)
    }
  }
}
