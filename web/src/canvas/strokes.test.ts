import { describe, expect, it } from 'vitest'
import { EMPTY_STROKES, type StrokesAction, strokesReducer, visibleStrokes } from './strokes'

const p = (x: number, y: number) => ({ x, y })
const run = (...actions: StrokesAction[]) => actions.reduce(strokesReducer, EMPTY_STROKES)

describe('strokesReducer', () => {
  it('builds strokes from start / move / end', () => {
    const state = run(
      { type: 'start', point: p(0, 0) },
      { type: 'move', point: p(1, 1) },
      { type: 'end' },
      { type: 'start', point: p(5, 5) },
      { type: 'end' },
    )
    expect(state.strokes).toEqual([[p(0, 0), p(1, 1)], [p(5, 5)]])
    expect(state.current).toBeNull()
  })

  it('exposes the stroke in progress', () => {
    const state = run({ type: 'start', point: p(0, 0) }, { type: 'move', point: p(2, 0) })
    expect(visibleStrokes(state)).toEqual([[p(0, 0), p(2, 0)]])
  })

  it('ignores duplicate points and moves without a stroke', () => {
    expect(run({ type: 'move', point: p(1, 1) })).toBe(EMPTY_STROKES)
    const state = run({ type: 'start', point: p(0, 0) }, { type: 'move', point: p(0, 0) })
    expect(state.current).toEqual([p(0, 0)])
  })

  it('undo removes the last stroke (or the one in progress), clear removes all', () => {
    const two = run(
      { type: 'start', point: p(0, 0) },
      { type: 'end' },
      { type: 'start', point: p(1, 1) },
      { type: 'end' },
    )
    expect(strokesReducer(two, { type: 'undo' }).strokes).toEqual([[p(0, 0)]])
    const drawing = strokesReducer(two, { type: 'start', point: p(9, 9) })
    expect(strokesReducer(drawing, { type: 'undo' }).strokes).toHaveLength(2)
    expect(strokesReducer(two, { type: 'clear' })).toBe(EMPTY_STROKES)
  })
})
