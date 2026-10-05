import { useCallback, useMemo, useReducer } from 'react'
import { DrawingCanvas } from './canvas/DrawingCanvas'
import { EMPTY_STROKES, strokesReducer, visibleStrokes } from './canvas/strokes'
import { GuessList } from './components/GuessList'
import { useClassifier } from './ml/useClassifier'
import { useLivePrediction } from './ml/useLivePrediction'

function App() {
  const [state, dispatch] = useReducer(strokesReducer, EMPTY_STROKES)
  const { status, result, classify, reset } = useClassifier()
  const strokes = useMemo(() => visibleStrokes(state), [state])
  const ready = status.state === 'ready'

  const classifyWhenReady = useCallback(
    (s: Parameters<typeof classify>[0]) => {
      if (ready) classify(s)
    },
    [ready, classify],
  )
  useLivePrediction(strokes, state.current !== null, classifyWhenReady, reset)

  return (
    <div className="min-h-screen bg-amber-50 text-slate-800">
      <main className="mx-auto flex max-w-4xl flex-col gap-6 px-4 py-8">
        <header>
          <h1 className="text-4xl font-bold tracking-tight">Doodle Live</h1>
          <p className="text-slate-600">Draw something. The model guesses after every stroke.</p>
        </header>

        <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_16rem]">
          <section className="flex flex-col gap-3">
            <DrawingCanvas state={state} dispatch={dispatch} />
            <div className="flex gap-2">
              <button
                type="button"
                className="rounded-lg bg-white px-4 py-2 text-sm font-medium shadow-sm hover:bg-slate-50 disabled:opacity-50"
                onClick={() => dispatch({ type: 'undo' })}
                disabled={strokes.length === 0}
              >
                Undo
              </button>
              <button
                type="button"
                className="rounded-lg bg-white px-4 py-2 text-sm font-medium shadow-sm hover:bg-slate-50 disabled:opacity-50"
                onClick={() => dispatch({ type: 'clear' })}
                disabled={strokes.length === 0}
              >
                Clear
              </button>
            </div>
          </section>

          <aside className="flex flex-col gap-3 rounded-2xl bg-white/70 p-4 shadow-sm">
            <h2 className="font-semibold">AI guesses</h2>
            {status.state === 'loading' && <p className="text-sm text-slate-500">Loading model…</p>}
            {status.state === 'error' && (
              <p role="alert" className="text-sm text-red-700">
                Could not load the model: {status.message}
              </p>
            )}
            {ready && <GuessList guesses={result?.top ?? []} />}
            {ready && result && (
              <p className="text-xs text-slate-400 tabular-nums">
                {result.inputLength} points · {result.ms.toFixed(1)} ms
              </p>
            )}
          </aside>
        </div>
      </main>
      <footer className="pb-6 text-center text-xs text-slate-500">
        Drawings from{' '}
        <a
          className="underline hover:text-slate-700"
          href="https://quickdraw.withgoogle.com/data"
          target="_blank"
          rel="noreferrer"
        >
          Quick, Draw!
        </a>{' '}
        (CC BY 4.0) ·{' '}
        <a
          className="underline hover:text-slate-700"
          href="https://github.com/MunzurAtak/doodle-live"
          target="_blank"
          rel="noreferrer"
        >
          GitHub
        </a>
      </footer>
    </div>
  )
}

export default App
