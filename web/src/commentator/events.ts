/**
 * Commentary events (SPEC §8.2). Pure functions: the caller keeps an `EventMemory` per
 * round and passes it back in, so "first", "changed" and "crossed for the first time"
 * can be decided without hidden state.
 */
import type { Guess } from '../ml/inference'

export type GameEventType =
  | 'ROUND_START'
  | 'FIRST_GUESS'
  | 'GUESS_CHANGED'
  | 'CONFIDENT'
  | 'STUCK'
  | 'ROUND_WON'
  | 'ROUND_LOST'

export const EVENT_PRIORITY: Record<GameEventType, 1 | 2 | 3> = {
  ROUND_START: 2,
  FIRST_GUESS: 1,
  GUESS_CHANGED: 1,
  CONFIDENT: 2,
  STUCK: 1,
  ROUND_WON: 3,
  ROUND_LOST: 3,
}

export const FIRST_GUESS_MIN = 0.15
export const GUESS_CHANGED_MIN = 0.25
export const CONFIDENT_MIN = 0.7
export const STUCK_AFTER_MS = 4000
export const STUCK_MAX_PROB = 0.5

export interface GameEvent {
  readonly type: GameEventType
  readonly priority: 1 | 2 | 3
  /** The guess the event is about (top-1 at the time). */
  readonly label?: string
  /** The target word; only set on ROUND_WON / ROUND_LOST. */
  readonly answer?: string
}

export interface EventMemory {
  readonly firstGuessMade: boolean
  readonly lastCommentedLabel: string | null
  readonly confidentLabels: readonly string[]
  /** Stroke count at which STUCK last fired (fires at most once per pause). */
  readonly stuckAtStrokes: number | null
}

export const EMPTY_MEMORY: EventMemory = {
  firstGuessMade: false,
  lastCommentedLabel: null,
  confidentLabels: [],
  stuckAtStrokes: null,
}

export const makeEvent = (type: GameEventType, extra: Partial<GameEvent> = {}): GameEvent => ({
  type,
  priority: EVENT_PRIORITY[type],
  ...extra,
})

/** Events caused by a new prediction. */
export function detectPredictionEvents(
  memory: EventMemory,
  top: readonly Guess[],
): { events: GameEvent[]; memory: EventMemory } {
  const best = top[0]
  if (!best) return { events: [], memory }
  const events: GameEvent[] = []
  let next = memory

  if (!memory.firstGuessMade) {
    if (best.prob >= FIRST_GUESS_MIN) {
      events.push(makeEvent('FIRST_GUESS', { label: best.label }))
      next = { ...next, firstGuessMade: true, lastCommentedLabel: best.label }
    }
  } else if (best.label !== memory.lastCommentedLabel && best.prob >= GUESS_CHANGED_MIN) {
    events.push(makeEvent('GUESS_CHANGED', { label: best.label }))
    next = { ...next, lastCommentedLabel: best.label }
  }

  if (best.prob >= CONFIDENT_MIN && !next.confidentLabels.includes(best.label)) {
    events.push(makeEvent('CONFIDENT', { label: best.label }))
    next = {
      ...next,
      lastCommentedLabel: best.label,
      confidentLabels: [...next.confidentLabels, best.label],
    }
  }
  return { events, memory: next }
}

/** STUCK: no new stroke for 4 s while drawing, and the model is still unsure. */
export function detectStuck(
  memory: EventMemory,
  ctx: { now: number; lastStrokeAt: number; strokes: number; topProb: number; label?: string },
): { events: GameEvent[]; memory: EventMemory } {
  const stalled = ctx.now - ctx.lastStrokeAt >= STUCK_AFTER_MS
  if (!stalled || ctx.topProb >= STUCK_MAX_PROB || memory.stuckAtStrokes === ctx.strokes) {
    return { events: [], memory }
  }
  return {
    events: [makeEvent('STUCK', { label: ctx.label })],
    memory: { ...memory, stuckAtStrokes: ctx.strokes },
  }
}
