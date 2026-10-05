/**
 * Scripted commentary (SPEC §8.7): used when the LLM is off, unavailable, or fails the
 * grounding check. `{label}` is the current guess, `{answer}` the target word.
 */
import type { GameEvent, GameEventType } from './events'

export const FALLBACK_LINES: Record<GameEventType, readonly string[]> = {
  ROUND_START: [
    'Ready when you are.',
    'New round, fresh eyes. Go!',
    "I'm watching every stroke.",
    'Surprise me.',
    'Pen down whenever you like.',
  ],
  FIRST_GUESS: [
    'Hmm... is that a {label}?',
    'My first thought is {label}.',
    'Could be a {label}?',
    'I want to say {label}, but keep going.',
    'Early days, but {label}?',
  ],
  GUESS_CHANGED: [
    'Wait, maybe a {label}?',
    'Oh, now it looks more like a {label}.',
    'Changing my mind: {label}?',
    'Hold on... {label}?',
    'Scratch that. A {label}?',
    "I'm leaning towards {label} now.",
  ],
  CONFIDENT: [
    "That's a {label}, I'm sure of it!",
    'Definitely a {label}!',
    'A {label}! Final answer.',
    'Now that is a {label}.',
    "It's a {label}, isn't it?",
  ],
  STUCK: [
    "I'm stumped. Add another detail?",
    'Give me a hint, one more stroke?',
    'Still thinking... draw a bit more.',
    'Hmm. Is it a {label}? I really am not sure.',
    'This one has me puzzled.',
  ],
  ROUND_WON: [
    'A {answer}! Got it.',
    'Yes! {answer}!',
    'Nailed it: {answer}.',
    'I knew it was a {answer}!',
    '{answer}! Great drawing.',
  ],
  ROUND_LOST: [
    'A {answer}? I never would have guessed.',
    'Out of time! So it was a {answer}.',
    'A {answer}... I see it now.',
    'That was a {answer}? You got me.',
    'Missed that one. A {answer}, of course.',
  ],
}

export function fillTemplate(template: string, event: GameEvent): string {
  const text = template
    .replaceAll('{label}', event.label ?? 'something')
    .replaceAll('{answer}', event.answer ?? 'mystery')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/** Pick a line for an event, avoiding the previous line when there is a choice. */
export function scriptedLine(
  event: GameEvent,
  previous: string | null = null,
  random: () => number = Math.random,
): string {
  const options = FALLBACK_LINES[event.type].map((t) => fillTemplate(t, event))
  const fresh = options.filter((line) => line !== previous)
  const pool = fresh.length > 0 ? fresh : options
  return pool[Math.floor(random() * pool.length)]
}
