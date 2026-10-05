import { useCallback, useEffect, useRef, useState } from 'react'
import { type ChatEngine, type LLMClient, WebLLMClient } from './llm'
import { type ModelSize, chooseModelId, detectWebGPU } from './models'

export type LLMStatus =
  | { readonly state: 'off' }
  | { readonly state: 'loading'; readonly progress: number; readonly text: string }
  | { readonly state: 'ready'; readonly modelId: string }
  | { readonly state: 'unsupported'; readonly reason: string }
  | { readonly state: 'error'; readonly message: string }

interface Engine extends ChatEngine {
  reload(modelId: string): Promise<void>
  unload(): Promise<void>
}

/**
 * Lazily loads WebLLM in a worker (SPEC §8.1). Nothing is downloaded until `wake()`;
 * the ~1 GB of weights are cached by the browser after the first time.
 */
export function useLLM() {
  const [status, setStatus] = useState<LLMStatus>({ state: 'off' })
  const [client, setClient] = useState<LLMClient | null>(null)
  const engine = useRef<Engine | null>(null)
  const worker = useRef<Worker | null>(null)
  const shaderF16 = useRef(true)

  const onProgress = useCallback((report: { progress: number; text: string }) => {
    setStatus({ state: 'loading', progress: report.progress, text: report.text })
  }, [])

  const wake = useCallback(
    async (size: ModelSize) => {
      setStatus({ state: 'loading', progress: 0, text: 'Checking for WebGPU…' })
      const gpu = await detectWebGPU()
      if (!gpu.ok) {
        setStatus({ state: 'unsupported', reason: gpu.reason })
        return
      }
      shaderF16.current = gpu.shaderF16
      const modelId = chooseModelId(size, gpu.shaderF16)
      try {
        if (engine.current) {
          setClient(null)
          await engine.current.reload(modelId)
        } else {
          // Loaded on demand so the main bundle stays small for people who never wake it.
          const webllm = await import('@mlc-ai/web-llm')
          worker.current = new Worker(new URL('./llm.worker.ts', import.meta.url), {
            type: 'module',
          })
          engine.current = (await webllm.CreateWebWorkerMLCEngine(worker.current, modelId, {
            initProgressCallback: onProgress,
          })) as unknown as Engine
        }
        setClient(new WebLLMClient(engine.current))
        setStatus({ state: 'ready', modelId })
      } catch (error) {
        setClient(null)
        setStatus({
          state: 'error',
          message: error instanceof Error ? error.message : String(error),
        })
      }
    },
    [onProgress],
  )

  useEffect(
    () => () => {
      void engine.current?.unload().catch(() => undefined)
      worker.current?.terminate()
    },
    [],
  )

  return { status, client, wake }
}
