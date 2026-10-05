export const ROUND_SECONDS = 20
export const COUNTDOWN_SECONDS = 3
export const ROUNDS_PER_GAME = 6
export const WIN_PROBABILITY = 0.5
export const QUICK_GUESS_STROKES = 3
export const QUICK_GUESS_BONUS = 50

/** Points for a won round (SPEC §7.4): 100 + 5 per full second left, +50 if guessed
 * within 3 strokes. A lost round scores 0. */
export function scoreRound(won: boolean, secondsLeft: number, strokes: number): number {
  if (!won) return 0
  const bonus = strokes <= QUICK_GUESS_STROKES ? QUICK_GUESS_BONUS : 0
  return 100 + 5 * Math.max(0, Math.floor(secondsLeft)) + bonus
}
