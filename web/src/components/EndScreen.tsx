import type { GameState } from '../game/gameReducer'

interface EndScreenProps {
  readonly game: GameState
  readonly onPlayAgain: () => void
  readonly onPractice: () => void
}

export function EndScreen({ game, onPlayAgain, onPractice }: EndScreenProps) {
  const won = game.results.filter((r) => r.won).length
  return (
    <section className="mx-auto flex w-full max-w-xl flex-col gap-6 rounded-[20px] bg-paper p-6 shadow-[0_1px_0_var(--color-rule),0_12px_32px_-16px_rgb(27_31_59/0.25)] sm:p-8">
      <div>
        <p className="text-ink-soft">Final score</p>
        <p className="font-display text-6xl font-bold tabular-nums">{game.score}</p>
        <p className="mt-1 text-ink-soft">
          The AI recognised {won} of your {game.results.length} drawings.
        </p>
      </div>
      <ol className="flex flex-col divide-y divide-rule">
        {game.results.map((r) => (
          <li key={r.word} className="flex items-baseline justify-between gap-4 py-2.5">
            <span className="font-display text-lg font-semibold">{r.word}</span>
            <span className="flex-1 text-sm text-ink-soft">
              {r.won
                ? `guessed after ${r.strokes} ${r.strokes === 1 ? 'stroke' : 'strokes'}`
                : r.outcome === 'skipped'
                  ? 'skipped'
                  : 'not guessed'}
            </span>
            <span className={`font-bold tabular-nums ${r.won ? 'text-hit' : 'text-ink-faint'}`}>
              {r.won ? `+${r.score}` : '0'}
            </span>
          </li>
        ))}
      </ol>
      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          autoFocus
          onClick={onPlayAgain}
          className="rounded-full bg-marker px-6 py-2.5 font-bold text-white hover:bg-marker-dark focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-marker"
        >
          Play again
        </button>
        <button
          type="button"
          onClick={onPractice}
          className="rounded-full border-2 border-ink px-6 py-2 font-bold hover:bg-board focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-marker"
        >
          Free drawing
        </button>
      </div>
    </section>
  )
}
