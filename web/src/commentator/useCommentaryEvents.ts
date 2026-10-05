import { useEffect, useReducer } from 'react'
import type { Phase } from '../game/gameReducer'
import type { Guess } from '../ml/inference'
import { type CommentaryState, INITIAL_COMMENTARY, commentaryReducer } from './commentary'

interface Inputs {
  readonly phase: Phase
  readonly round: number
  readonly word: string | null
  readonly top: readonly Guess[]
  readonly strokes: number
  readonly now: number
}

/** Wires game state into the pure event reducer. */
export function useCommentaryEvents({
  phase,
  round,
  word,
  top,
  strokes,
  now,
}: Inputs): CommentaryState {
  const [state, dispatch] = useReducer(commentaryReducer, INITIAL_COMMENTARY)

  useEffect(() => {
    dispatch({ type: 'phase', phase, word, label: top[0]?.label, now: performance.now() })
    // React to phase changes and new rounds only, not to every new guess.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, round])

  useEffect(() => {
    if (phase === 'drawing' || phase === 'idle') dispatch({ type: 'prediction', top })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [top])

  useEffect(() => {
    dispatch({ type: 'strokes', strokes, now: performance.now() })
  }, [strokes])

  useEffect(() => {
    if (phase === 'drawing') dispatch({ type: 'tick', now, top })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [now])

  return state
}
