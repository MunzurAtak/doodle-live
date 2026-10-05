import { ROUND_SECONDS } from '../game/scoring'

export function Timer({ seconds }: { readonly seconds: number }) {
  const urgent = seconds <= 5
  return (
    <div
      className="flex w-24 flex-col items-end gap-1"
      aria-label={`${Math.ceil(seconds)} seconds left`}
    >
      <span
        className={`font-display text-3xl font-bold tabular-nums ${urgent ? 'text-miss' : 'text-ink'}`}
      >
        {Math.ceil(seconds)}s
      </span>
      <div className="h-1.5 w-full rounded-full bg-rule">
        <div
          className={`h-1.5 rounded-full ${urgent ? 'bg-miss' : 'bg-marker'}`}
          style={{ width: `${(seconds / ROUND_SECONDS) * 100}%` }}
        />
      </div>
    </div>
  )
}
