import { useEffect, useRef } from 'react'
import type { Point } from './preprocess'

export const LIVE_THROTTLE_MS = 250

/**
 * Ask for a prediction whenever the drawing changes: immediately when no stroke is in
 * progress (pen-up, undo, clear) and at most every `throttleMs` while a stroke is drawn.
 */
export function useLivePrediction(
  strokes: readonly (readonly Point[])[],
  inProgress: boolean,
  classify: (strokes: readonly (readonly Point[])[], final: boolean) => void,
  onEmpty: () => void,
  throttleMs = LIVE_THROTTLE_MS,
) {
  const lastSent = useRef(-Infinity)

  useEffect(() => {
    if (strokes.length === 0) {
      onEmpty()
      return
    }
    const now = performance.now()
    if (!inProgress || now - lastSent.current >= throttleMs) {
      lastSent.current = now
      classify(strokes, !inProgress)
    }
  }, [strokes, inProgress, classify, onEmpty, throttleMs])
}
