import { useEffect, useReducer, useState } from 'react'
import { INITIAL_GAME, gameReducer } from './gameReducer'

/** Game state plus a clock: dispatches `tick` and re-renders ~10x per second while a
 * countdown or round is running. */
export function useGame() {
  const [game, dispatch] = useReducer(gameReducer, INITIAL_GAME)
  const [now, setNow] = useState(() => performance.now())
  const running = game.phase === 'countdown' || game.phase === 'drawing'

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => {
      const t = performance.now()
      setNow(t)
      dispatch({ type: 'tick', now: t })
    }, 100)
    return () => window.clearInterval(id)
  }, [running])

  return { game, dispatch, now }
}
