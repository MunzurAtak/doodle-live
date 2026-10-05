/**
 * Prompt construction (SPEC §8.4). Pure functions.
 *
 * The model never sees the target word while the user is drawing: `answer` is only
 * included for ROUND_WON / ROUND_LOST, whatever the caller passes in.
 */
import type { Guess } from '../ml/inference'
import type { GameEvent, GameEventType } from './events'
import type { Personality } from './personalities'

export interface ChatMessage {
  readonly role: 'system' | 'user' | 'assistant'
  readonly content: string
}

export type Confidence = 'high' | 'medium' | 'low'

export interface PromptContext {
  readonly strokes: number
  readonly secondsLeft: number | null
  /** Current top guesses from the classifier (only the first 3 are used). */
  readonly candidates: readonly Guess[]
  /** This round's previous lines, oldest first (only the last 3 are used). */
  readonly previousLines: readonly string[]
  /** The target word. Ignored unless the event ends the round. */
  readonly answer: string | null
}

export const GENERATION = { temperature: 0.8, max_tokens: 48 } as const

export const SHARED_RULES = `You are an AI contestant on a drawing game show. Someone is drawing and you guess out loud what it is.
Rules:
- You cannot see the drawing. You only know the candidate list your vision model gives you.
- Only mention objects from the candidate list (or the answer, when given). Never invent other objects.
- Reply with one or two short sentences, at most 25 words. No emojis, no lists, no numbers or percentages.
- Match your certainty to the confidence words: hesitate when low, be excited when high.
- Do not repeat your previous lines.`

const TASKS: Record<GameEventType, string> = {
  ROUND_START:
    'A new round starts and nothing is drawn yet. Say something short to get ready. Do not guess.',
  FIRST_GUESS: 'Share your first guess.',
  GUESS_CHANGED: 'Your top guess changed. React to it.',
  CONFIDENT: 'You are now confident. Say your guess with conviction.',
  STUCK: 'The drawer paused and you are unsure. Ask for more detail.',
  ROUND_WON: 'You guessed it. The correct answer is "answer". Celebrate briefly.',
  ROUND_LOST: 'Time ran out before you guessed it. The correct answer is "answer". React to it.',
}

export function confidenceWord(prob: number): Confidence {
  if (prob >= 0.6) return 'high'
  if (prob >= 0.3) return 'medium'
  return 'low'
}

export const endsRound = (type: GameEventType) => type === 'ROUND_WON' || type === 'ROUND_LOST'

export function buildUserState(event: GameEvent, ctx: PromptContext): Record<string, unknown> {
  const state: Record<string, unknown> = {
    event: event.type,
    task: TASKS[event.type],
    strokes: ctx.strokes,
  }
  if (ctx.secondsLeft !== null) state.seconds_left = Math.ceil(ctx.secondsLeft)
  state.candidates = ctx.candidates.slice(0, 3).map((g) => ({
    label: g.label,
    confidence: confidenceWord(g.prob),
  }))
  state.previous_lines = ctx.previousLines.slice(-3)
  if (endsRound(event.type) && ctx.answer) state.answer = ctx.answer
  return state
}

/** Stateless chat request: system rules + personality, then the game state as JSON. */
export function buildMessages(
  event: GameEvent,
  ctx: PromptContext,
  personality: Personality,
): ChatMessage[] {
  return [
    { role: 'system', content: `${SHARED_RULES}\n\nPersonality: ${personality.prompt}` },
    { role: 'user', content: JSON.stringify(buildUserState(event, ctx)) },
  ]
}

/** Small models sometimes ignore length rules; `max_tokens` cuts them off mid-sentence.
 * Keep everything up to the last complete sentence. */
export function trimToSentence(text: string): string {
  const clean = text.replace(/\s+/g, ' ').trim()
  const match = clean.match(/^.*[.!?](?=["')\]]?(\s|$))["')\]]?/)
  return match && match[0].length >= 3 ? match[0] : clean
}
