import type { Mood } from '../commentator/commentary'

/** The AI contestant: a small screen with eyes that react to its own confidence. */
export function Avatar({ mood }: { readonly mood: Mood }) {
  return (
    <svg viewBox="0 0 64 64" className="h-14 w-14 shrink-0" role="img" aria-label={`AI, ${mood}`}>
      <rect x="4" y="8" width="56" height="44" rx="14" fill="var(--color-ink)" />
      <rect x="26" y="52" width="12" height="6" rx="2" fill="var(--color-ink)" />
      <g fill="none" stroke="var(--color-highlight)" strokeWidth="4" strokeLinecap="round">
        {mood === 'happy' && (
          <>
            <path d="M18 32 q5 -8 10 0" />
            <path d="M36 32 q5 -8 10 0" />
          </>
        )}
        {mood === 'sad' && (
          <>
            <path d="M18 28 q5 6 10 0" />
            <path d="M36 28 q5 6 10 0" />
          </>
        )}
      </g>
      {(mood === 'idle' || mood === 'thinking' || mood === 'sure') && (
        <g fill="var(--color-highlight)">
          <circle
            cx={mood === 'thinking' ? 25 : 23}
            cy={mood === 'thinking' ? 26 : 30}
            r={mood === 'sure' ? 6 : 4}
          />
          <circle
            cx={mood === 'thinking' ? 43 : 41}
            cy={mood === 'thinking' ? 26 : 30}
            r={mood === 'sure' ? 6 : 4}
          />
        </g>
      )}
    </svg>
  )
}
