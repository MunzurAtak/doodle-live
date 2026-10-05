import type { Point } from '../ml/preprocess'

/** Drawing state: finished strokes plus the stroke currently being drawn (if any). */
export interface StrokesState {
  readonly strokes: readonly (readonly Point[])[]
  readonly current: readonly Point[] | null
}

export type StrokesAction =
  | { readonly type: 'start'; readonly point: Point }
  | { readonly type: 'move'; readonly point: Point }
  | { readonly type: 'end' }
  | { readonly type: 'undo' }
  | { readonly type: 'clear' }

export const EMPTY_STROKES: StrokesState = { strokes: [], current: null }

export function strokesReducer(state: StrokesState, action: StrokesAction): StrokesState {
  switch (action.type) {
    case 'start':
      return { strokes: commit(state), current: [action.point] }
    case 'move': {
      if (!state.current) return state
      const last = state.current[state.current.length - 1]
      if (last.x === action.point.x && last.y === action.point.y) return state
      return { ...state, current: [...state.current, action.point] }
    }
    case 'end':
      return state.current ? { strokes: commit(state), current: null } : state
    case 'undo':
      return state.current
        ? { strokes: state.strokes, current: null }
        : { strokes: state.strokes.slice(0, -1), current: null }
    case 'clear':
      return EMPTY_STROKES
  }
}

function commit(state: StrokesState): readonly (readonly Point[])[] {
  return state.current && state.current.length > 0
    ? [...state.strokes, state.current]
    : state.strokes
}

/** Everything drawn so far, including the stroke in progress. */
export function visibleStrokes(state: StrokesState): readonly (readonly Point[])[] {
  return state.current ? [...state.strokes, state.current] : state.strokes
}
