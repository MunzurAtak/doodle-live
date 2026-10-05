import type { GameState } from '../game/gameReducer'
import { COUNTDOWN_SECONDS, ROUNDS_PER_GAME } from '../game/scoring'

interface RoundOverlayProps {
  readonly game: GameState
  readonly now: number
  readonly onNext: () => void
}

/** Covers the canvas during the countdown and between rounds. */
export function RoundOverlay({ game, now, onNext }: RoundOverlayProps) {
  if (game.phase === 'countdown') {
    const count = Math.max(1, Math.ceil((game.deadline - now) / 1000))
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-[20px] bg-paper/85">
        <p className="text-ink-soft">
          Round {game.round + 1} of {ROUNDS_PER_GAME}. Your word is
        </p>
        <p className="font-display text-5xl font-bold">
          <span className="highlight">{game.words[game.round]}</span>
        </p>
        <p
          key={count}
          className="count-pop mt-4 font-display text-7xl font-bold text-marker"
          aria-live="assertive"
        >
          {count <= COUNTDOWN_SECONDS ? count : ''}
        </p>
      </div>
    )
  }
  if (game.phase !== 'roundWon' && game.phase !== 'roundLost') return null
  const result = game.results[game.results.length - 1]
  const last = game.round + 1 >= ROUNDS_PER_GAME
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 rounded-[20px] bg-paper/90 px-6 text-center">
      <p className={`font-display text-4xl font-bold ${result.won ? 'text-hit' : 'text-miss'}`}>
        {result.outcome === 'guessed'
          ? 'Guessed it'
          : result.outcome === 'skipped'
            ? 'Skipped'
            : "Time's up"}
      </p>
      <p className="max-w-xs text-ink-soft">
        {result.won
          ? `The AI recognised your ${result.word} after ${result.strokes} ${result.strokes === 1 ? 'stroke' : 'strokes'}, with ${result.secondsLeft}s to spare.`
          : result.outcome === 'skipped'
            ? `On to the next word. That was ${article(result.word)} ${result.word}.`
            : `The AI didn't recognise your ${result.word} this time.`}
      </p>
      {result.won && <p className="font-display text-2xl font-bold">+{result.score}</p>}
      <button
        type="button"
        autoFocus
        onClick={onNext}
        className="mt-2 rounded-full bg-marker px-6 py-2.5 font-bold text-white hover:bg-marker-dark focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-marker"
      >
        {last ? 'See results' : 'Next word'}
      </button>
    </div>
  )
}

const article = (word: string) => (/^[aeiou]/i.test(word) ? 'an' : 'a')
