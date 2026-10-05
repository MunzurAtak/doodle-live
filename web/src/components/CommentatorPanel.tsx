import type { LLMStatus } from '../commentator/useLLM'
import { MODEL_OPTIONS, type ModelSize } from '../commentator/models'
import { PERSONALITIES, type PersonalityId } from '../commentator/personalities'

export interface CommentatorSettings {
  readonly enabled: boolean
  readonly voice: boolean
  readonly personality: PersonalityId
  readonly modelSize: ModelSize
}

interface CommentatorPanelProps {
  readonly status: LLMStatus
  readonly settings: CommentatorSettings
  readonly onChange: (settings: CommentatorSettings) => void
  readonly onWake: (size: ModelSize) => void
}

const smallButton =
  'rounded-full border-2 border-ink px-4 py-1.5 text-sm font-bold hover:bg-board focus-visible:outline-3 focus-visible:outline-offset-2 focus-visible:outline-marker'

function Status({ status, settings, onWake }: Omit<CommentatorPanelProps, 'onChange'>) {
  const model = MODEL_OPTIONS[settings.modelSize]
  switch (status.state) {
    case 'off':
      return (
        <div className="flex flex-col items-start gap-2 text-sm text-ink-soft">
          <p>
            Right now I read from a script. Wake up the language model ({model.label}) to hear what
            I actually think. It downloads once ({model.download}) and runs on your GPU.
          </p>
          <button type="button" className={smallButton} onClick={() => onWake(settings.modelSize)}>
            Wake up the commentator
          </button>
        </div>
      )
    case 'loading':
      return (
        <div className="flex flex-col gap-1.5 text-sm text-ink-soft">
          <div
            className="h-2 rounded-full bg-rule"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(status.progress * 100)}
            aria-label="Loading the language model"
          >
            <div
              className="h-2 rounded-full bg-marker transition-[width] motion-reduce:transition-none"
              style={{ width: `${Math.max(2, status.progress * 100)}%` }}
            />
          </div>
          <p className="line-clamp-2">{status.text || 'Loading…'}</p>
        </div>
      )
    case 'ready':
      return (
        <p className="text-sm text-ink-soft">
          {status.modelId.split('-Instruct')[0].replace('-', ' ')} is commenting, live on your GPU.
          The game still only tells it what the vision model sees.
        </p>
      )
    case 'unsupported':
      return (
        <p className="text-sm text-ink-soft" role="status">
          {status.reason} The AI voice needs WebGPU, for example in a recent Chrome or Edge, so I'm
          using scripted lines. The game works the same.
        </p>
      )
    case 'error':
      return (
        <div className="flex flex-col items-start gap-2 text-sm">
          <p role="alert" className="text-miss">
            The language model could not start: {status.message}
          </p>
          <button type="button" className={smallButton} onClick={() => onWake(settings.modelSize)}>
            Try again
          </button>
        </div>
      )
  }
}

export function CommentatorPanel({ status, settings, onChange, onWake }: CommentatorPanelProps) {
  const set = (patch: Partial<CommentatorSettings>) => onChange({ ...settings, ...patch })
  const loadedSize =
    status.state === 'ready' ? (status.modelId.includes('0.5B') ? 'light' : 'default') : null
  return (
    <div className="flex flex-col gap-3">
      <Status status={status} settings={settings} onWake={onWake} />
      <details className="text-sm">
        <summary className="cursor-pointer font-bold hover:text-marker">
          Commentator settings
        </summary>
        <div className="mt-3 flex flex-col gap-3">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              className="h-4 w-4 accent-marker"
              checked={settings.enabled}
              onChange={(e) => set({ enabled: e.target.checked })}
            />
            Commentary
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              className="h-4 w-4 accent-marker"
              checked={settings.voice}
              onChange={(e) => set({ voice: e.target.checked })}
            />
            Read lines aloud
          </label>
          <label className="flex flex-col gap-1">
            Personality
            <select
              className="rounded-lg border-2 border-ink bg-paper px-2 py-1.5"
              value={settings.personality}
              onChange={(e) => set({ personality: e.target.value as PersonalityId })}
            >
              {PERSONALITIES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <fieldset className="flex flex-col gap-1">
            <legend className="mb-1">Language model</legend>
            {(Object.keys(MODEL_OPTIONS) as ModelSize[]).map((size) => (
              <label key={size} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="model-size"
                  className="h-4 w-4 accent-marker"
                  checked={settings.modelSize === size}
                  onChange={() => set({ modelSize: size })}
                />
                {`${MODEL_OPTIONS[size].label} (${MODEL_OPTIONS[size].download})${size === 'light' ? ', faster, sillier' : ''}`}
              </label>
            ))}
            {loadedSize && loadedSize !== settings.modelSize && (
              <button
                type="button"
                className={`${smallButton} mt-1 self-start`}
                onClick={() => onWake(settings.modelSize)}
              >
                Switch to {MODEL_OPTIONS[settings.modelSize].label}
              </button>
            )}
          </fieldset>
        </div>
      </details>
    </div>
  )
}
