import type { StrokeSnapshot } from '../game/gameReducer'
import { WIN_PROBABILITY } from '../game/scoring'

interface ConfidenceChartProps {
  readonly history: readonly StrokeSnapshot[]
  readonly categories: readonly string[]
  /** Target word: drawn as the blue line. Without a target, the current top-1 is. */
  readonly target?: string | null
}

const W = 320
const H = 168
const PAD = { left: 38, right: 74, top: 10, bottom: 22 }
const FOCUS = '#2a78d6'
const OTHER = '#9aa0b8'

/** Probability of the target and the current top guesses after every stroke (SVG). */
export function ConfidenceChart({ history, categories, target = null }: ConfidenceChartProps) {
  if (history.length === 0) {
    return <p className="text-sm text-ink-soft">The chart fills in stroke by stroke.</p>
  }
  const latest = history[history.length - 1].probs
  const ranked = latest.map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p)
  const focusIndex = target ? categories.indexOf(target) : ranked[0].i
  const others = ranked.filter((r) => r.i !== focusIndex).slice(0, target ? 2 : 2)
  const series = [
    { index: focusIndex, focus: true },
    ...others.map((r) => ({ index: r.i, focus: false })),
  ].filter((s) => s.index >= 0)

  const maxStroke = Math.max(5, history[history.length - 1].strokes)
  const x = (stroke: number) =>
    PAD.left + ((stroke - 1) / Math.max(1, maxStroke - 1)) * (W - PAD.left - PAD.right)
  const y = (p: number) => PAD.top + (1 - p) * (H - PAD.top - PAD.bottom)

  // Direct labels at the line ends, nudged apart so they never overlap.
  const labels = series.map((s) => ({ ...s, y: y(latest[s.index]) })).sort((a, b) => a.y - b.y)
  const GAP = 13
  for (let k = 1; k < labels.length; k++) labels[k].y = Math.max(labels[k].y, labels[k - 1].y + GAP)
  // If the stack runs below the plot, push it back up.
  const floor = H - PAD.bottom
  for (let k = labels.length - 1; k >= 0; k--) {
    const limit = k === labels.length - 1 ? floor : labels[k + 1].y - GAP
    labels[k].y = Math.min(labels[k].y, limit)
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Confidence per stroke">
      {[0, 0.5, 1].map((p) => (
        <g key={p}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(p)} y2={y(p)} stroke="var(--color-rule)" />
          <text
            x={PAD.left - 6}
            y={y(p) + 4}
            textAnchor="end"
            fontSize="10"
            fill="var(--color-ink-soft)"
          >
            {p * 100}%
          </text>
        </g>
      ))}
      {target && (
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={y(WIN_PROBABILITY)}
          y2={y(WIN_PROBABILITY)}
          stroke={FOCUS}
          strokeDasharray="4 4"
          opacity={0.6}
        />
      )}
      {Array.from({ length: maxStroke }, (_, k) => k + 1).map((stroke) => (
        <text
          key={stroke}
          x={x(stroke)}
          y={H - 8}
          textAnchor="middle"
          fontSize="10"
          fill="var(--color-ink-soft)"
        >
          {stroke}
        </text>
      ))}
      {[...series].reverse().map(({ index, focus }) => {
        const color = focus ? FOCUS : OTHER
        const points = history.map((s) => `${x(s.strokes)},${y(s.probs[index])}`).join(' ')
        return (
          <g key={index}>
            <polyline
              points={points}
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeLinejoin="round"
            />
            {history.map((s) => (
              <circle
                key={s.strokes}
                cx={x(s.strokes)}
                cy={y(s.probs[index])}
                r={4}
                fill={color}
                stroke="var(--color-paper)"
                strokeWidth={2}
              >
                <title>{`Stroke ${s.strokes}: ${categories[index]} ${Math.round(s.probs[index] * 100)}%`}</title>
              </circle>
            ))}
          </g>
        )
      })}
      {labels.map((l) => (
        <text
          key={l.index}
          x={W - PAD.right + 8}
          y={l.y + 4}
          fontSize="11"
          fontWeight={l.focus ? 700 : 400}
          fill={l.focus ? 'var(--color-ink)' : 'var(--color-ink-soft)'}
        >
          {categories[l.index]}
        </text>
      ))}
    </svg>
  )
}
