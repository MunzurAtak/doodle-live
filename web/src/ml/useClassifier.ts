import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ClassifierClient,
  type ClassifierResult,
  type WorkerLike,
  createClassifierWorker,
} from './classifier'
import { modelsBaseUrl } from './meta'
import type { Point } from './preprocess'

export type ClassifierStatus =
  | { readonly state: 'loading' }
  | { readonly state: 'ready' }
  | { readonly state: 'error'; readonly message: string }

/** Owns the classifier worker for the lifetime of the component. */
export function useClassifier(createWorker: () => WorkerLike = createClassifierWorker) {
  const clientRef = useRef<ClassifierClient | null>(null)
  const [status, setStatus] = useState<ClassifierStatus>(() =>
    typeof Worker === 'undefined'
      ? { state: 'error', message: 'This browser does not support Web Workers.' }
      : { state: 'loading' },
  )
  const [result, setResult] = useState<ClassifierResult | null>(null)

  useEffect(() => {
    if (typeof Worker === 'undefined') return
    const client = new ClassifierClient(
      createWorker(),
      modelsBaseUrl(new URL(import.meta.env.BASE_URL, window.location.href).href),
      {
        onReady: () => setStatus({ state: 'ready' }),
        onResult: setResult,
        onError: (message) => setStatus({ state: 'error', message }),
      },
    )
    clientRef.current = client
    return () => {
      client.dispose()
      clientRef.current = null
    }
  }, [createWorker])

  const classify = useCallback((strokes: readonly (readonly Point[])[]) => {
    clientRef.current?.classify(strokes)
  }, [])

  const reset = useCallback(() => setResult(null), [])

  return { status, result, classify, reset }
}
