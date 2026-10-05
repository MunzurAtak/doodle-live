import { useCallback, useEffect, useMemo, useReducer } from 'react'
import { DrawingCanvas } from './canvas/DrawingCanvas'
import { EMPTY_STROKES, strokesReducer, visibleStrokes } from './canvas/strokes'
import { useScriptedCommentary } from './commentator/useScriptedCommentary'
import { Avatar } from './components/Avatar'
import { CategoryList } from './components/CategoryList'
import { ConfidenceChart } from './components/ConfidenceChart'
import { EndScreen } from './components/EndScreen'
import { GuessList } from './components/GuessList'
import { RoundOverlay } from './components/RoundOverlay'
import { SpeechBubble } from './components/SpeechBubble'
import { Timer } from './components/Timer'
import { currentWord, secondsLeft } from './game/gameReducer'
import { ROUNDS_PER_GAME } from './game/scoring'
import { useGame } from './game/useGame'
import { pickWords } from './game/words'
import type { Point } from './ml/preprocess'
import { useClassifier } from './ml/useClassifier'
import { useLivePrediction } from './ml/useLivePrediction'

const button =
  'rounded-full border-2 border-ink px-4 py-1.5 text-sm font-bold hover:bg-board disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-marker'
const primaryButton =
  'rounded-full bg-marker px-5 py-2 font-bold text-white hover:bg-marker-dark disabled:opacity-40 focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-marker'

function App() {
  const [drawing, draw] = useReducer(strokesReducer, EMPTY_STROKES)
  const { status, result, categories, classify, reset } = useClassifier()
  const { game, dispatch, now } = useGame()
  const strokes = useMemo(() => visibleStrokes(drawing), [drawing])
  const ready = status.state === 'ready'
  const inGame = game.phase !== 'idle'
  const canDraw = ready && (game.phase === 'idle' || game.phase === 'drawing')
  const word = currentWord(game)

  const classifyWhenReady = useCallback(
    (s: readonly (readonly Point[])[], final: boolean) => {
      if (ready) classify(s, final)
    },
    [ready, classify],
  )
  useLivePrediction(strokes, drawing.current !== null, classifyWhenReady, reset)

  // Feed every classifier result into the game (win check, per-stroke history).
  useEffect(() => {
    if (!result) return
    dispatch({
      type: 'prediction',
      now: performance.now(),
      label: result.top[0]?.label ?? null,
      prob: result.top[0]?.prob ?? 0,
      probs: result.probs,
      strokes: result.strokes,
      final: result.final,
    })
  }, [result, dispatch])

  // Fresh canvas whenever a countdown starts or the game ends.
  useEffect(() => {
    if (game.phase === 'countdown' || game.phase === 'gameOver') draw({ type: 'clear' })
  }, [game.phase, game.round])

  const top = useMemo(
    () => (strokes.length > 0 && result ? result.top : []),
    [strokes.length, result],
  )
  const commentary = useScriptedCommentary({
    phase: game.phase,
    round: game.round,
    word,
    top,
    strokes: strokes.length,
    now,
  })

  const start = () =>
    dispatch({
      type: 'start',
      words: pickWords(categories, ROUNDS_PER_GAME),
      now: performance.now(),
    })

  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 px-4 pt-6 sm:px-6">
        <h1 className="font-display text-3xl font-extrabold tracking-tight">Doodle Live</h1>
        {inGame && game.phase !== 'gameOver' ? (
          <dl className="flex gap-6 text-right">
            <div>
              <dt className="text-xs text-ink-soft">Round</dt>
              <dd className="font-display text-xl font-bold tabular-nums">
                {game.round + 1}/{ROUNDS_PER_GAME}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-ink-soft">Score</dt>
              <dd className="font-display text-xl font-bold tabular-nums">{game.score}</dd>
            </div>
          </dl>
        ) : (
          game.phase === 'idle' && (
            <button type="button" className={primaryButton} onClick={start} disabled={!ready}>
              Start game
            </button>
          )
        )}
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6">
        {game.phase === 'gameOver' ? (
          <EndScreen
            game={game}
            onPlayAgain={start}
            onPractice={() => dispatch({ type: 'quit' })}
          />
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <section className="mx-auto flex w-full max-w-[38rem] flex-col gap-3">
              <div className="flex min-h-14 items-end justify-between gap-4">
                {game.phase === 'idle' ? (
                  <p className="max-w-md text-ink-soft">
                    Free drawing. Sketch anything and watch the AI guess, or start a game: 6 words,
                    20 seconds each.
                  </p>
                ) : game.phase === 'countdown' ? (
                  <p className="text-ink-soft">Get ready…</p>
                ) : (
                  <p className="font-display text-4xl font-bold sm:text-5xl">
                    <span className="sr-only">Draw: </span>
                    <span className="highlight">{word}</span>
                  </p>
                )}
                {game.phase === 'drawing' && <Timer seconds={secondsLeft(game, now)} />}
              </div>

              <div className="relative">
                <div className="paper rounded-[20px] shadow-[0_1px_0_var(--color-rule),0_12px_32px_-16px_rgb(27_31_59/0.25)]">
                  <DrawingCanvas
                    state={drawing}
                    dispatch={draw}
                    disabled={!canDraw}
                    strokeColor="#1b1f3b"
                  />
                </div>
                <RoundOverlay
                  game={game}
                  now={now}
                  onNext={() => dispatch({ type: 'next', now: performance.now() })}
                />
                {status.state === 'loading' && (
                  <p className="absolute inset-0 flex items-center justify-center text-ink-soft">
                    Loading the model…
                  </p>
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className={button}
                  onClick={() => draw({ type: 'undo' })}
                  disabled={!canDraw || strokes.length === 0}
                >
                  Undo
                </button>
                <button
                  type="button"
                  className={button}
                  onClick={() => draw({ type: 'clear' })}
                  disabled={!canDraw || strokes.length === 0}
                >
                  Clear
                </button>
                {game.phase === 'drawing' && (
                  <button
                    type="button"
                    className={button}
                    onClick={() => dispatch({ type: 'skip', now: performance.now() })}
                  >
                    Skip word
                  </button>
                )}
                {inGame && (
                  <button
                    type="button"
                    className={`${button} ml-auto`}
                    onClick={() => dispatch({ type: 'quit' })}
                  >
                    Quit game
                  </button>
                )}
              </div>
            </section>

            <aside
              className="flex flex-col gap-5 rounded-[20px] border-2 border-ink bg-paper/60 p-5"
              aria-label="AI contestant"
            >
              <div className="flex items-start gap-3">
                <Avatar mood={commentary.mood} />
                <SpeechBubble
                  text={commentary.line}
                  placeholder={ready ? 'Draw something and I will guess out loud.' : 'Warming up…'}
                />
              </div>

              {status.state === 'error' && (
                <p role="alert" className="text-sm text-miss">
                  The model could not be loaded: {status.message}. Try reloading the page.
                </p>
              )}

              <div className="flex flex-col gap-2">
                <h2 className="font-display text-lg font-bold">Top guesses</h2>
                <GuessList guesses={top} target={word} />
              </div>

              <div className="flex flex-col gap-2">
                <h2 className="font-display text-lg font-bold">Confidence per stroke</h2>
                <ConfidenceChart history={game.history} categories={categories} target={word} />
              </div>

              {game.phase === 'idle' && categories.length > 0 && (
                <CategoryList categories={categories} />
              )}
            </aside>
          </div>
        )}
      </main>

      <footer className="px-4 pb-6 text-center text-sm text-ink-soft">
        Trained on Google's{' '}
        <a
          className="underline hover:text-ink"
          href="https://quickdraw.withgoogle.com/data"
          target="_blank"
          rel="noreferrer"
        >
          Quick, Draw! dataset
        </a>{' '}
        (CC BY 4.0). Everything runs in your browser.{' '}
        <a
          className="underline hover:text-ink"
          href="https://github.com/MunzurAtak/doodle-live"
          target="_blank"
          rel="noreferrer"
        >
          Source on GitHub
        </a>
      </footer>
    </div>
  )
}

export default App
