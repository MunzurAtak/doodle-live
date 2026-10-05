import type { Guess } from '../ml/inference'

interface GuessListProps {
  readonly guesses: readonly Guess[]
}

export function GuessList({ guesses }: GuessListProps) {
  if (guesses.length === 0) {
    return <p className="text-sm text-slate-500">Start drawing and I'll start guessing.</p>
  }
  return (
    <ol className="flex flex-col gap-2" aria-label="Top guesses">
      {guesses.map((guess, i) => {
        const percent = Math.round(guess.prob * 100)
        return (
          <li key={guess.label} className="flex flex-col gap-1">
            <div className="flex justify-between text-sm">
              <span className={i === 0 ? 'font-semibold text-slate-900' : 'text-slate-700'}>
                {guess.label}
              </span>
              <span className="text-slate-500 tabular-nums">{percent}%</span>
            </div>
            <div className="h-2 rounded-full bg-slate-200">
              <div
                className={`h-2 rounded-full transition-[width] duration-200 ${i === 0 ? 'bg-indigo-500' : 'bg-indigo-300'}`}
                style={{ width: `${Math.max(percent, 1)}%` }}
              />
            </div>
          </li>
        )
      })}
    </ol>
  )
}
