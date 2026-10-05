import type { Guess } from '../ml/inference'

interface GuessListProps {
  readonly guesses: readonly Guess[]
  readonly target?: string | null
}

export function GuessList({ guesses, target = null }: GuessListProps) {
  if (guesses.length === 0) {
    return <p className="text-sm text-ink-soft">Guesses appear after your first stroke.</p>
  }
  return (
    <ol className="flex flex-col gap-2" aria-label="Top guesses">
      {guesses.map((guess, i) => {
        const percent = Math.round(guess.prob * 100)
        const isTarget = guess.label === target
        return (
          <li
            key={guess.label}
            className="grid grid-cols-[7rem_1fr_2.5rem] items-center gap-2 text-sm"
          >
            <span className={`truncate ${i === 0 ? 'font-bold' : ''}`}>{guess.label}</span>
            <div className="h-2.5 rounded-full bg-rule">
              <div
                className={`h-2.5 rounded-full transition-[width] duration-200 motion-reduce:transition-none ${
                  isTarget ? 'bg-marker' : i === 0 ? 'bg-ink' : 'bg-ink-faint'
                }`}
                style={{ width: `${Math.max(percent, 2)}%` }}
              />
            </div>
            <span className="text-right text-ink-soft tabular-nums">{percent}%</span>
          </li>
        )
      })}
    </ol>
  )
}
