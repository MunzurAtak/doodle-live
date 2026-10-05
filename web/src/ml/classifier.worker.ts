/// <reference lib="webworker" />
/**
 * Classifier Web Worker: loads the ONNX model once, then preprocesses and classifies
 * drawings off the main thread. If several requests arrive while one is running, only
 * the newest is processed (older ones would be stale by the time they finish).
 */
import * as ort from 'onnxruntime-web/wasm'
import { classify } from './inference'
import { type ModelMeta, parseModelMeta } from './meta'
import type { WorkerRequest, WorkerResponse } from './protocol'

// GitHub Pages can't send the COOP/COEP headers multi-threaded WASM needs.
ort.env.wasm.numThreads = 1

declare const self: DedicatedWorkerGlobalScope

type ClassifyRequest = Extract<WorkerRequest, { type: 'classify' }>

let model: { session: ort.InferenceSession; meta: ModelMeta } | null = null
let busy = false
let pending: ClassifyRequest | null = null

const post = (message: WorkerResponse) => self.postMessage(message)

async function init(modelsUrl: string): Promise<void> {
  try {
    const metaResponse = await fetch(new URL('model_meta.json', modelsUrl))
    if (!metaResponse.ok) throw new Error(`model_meta.json: HTTP ${metaResponse.status}`)
    const meta = parseModelMeta(await metaResponse.json())
    const session = await ort.InferenceSession.create(new URL('classifier.onnx', modelsUrl).href, {
      executionProviders: ['wasm'],
    })
    model = { session, meta }
    post({ type: 'ready', categories: meta.categories })
  } catch (error) {
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}

async function drain(): Promise<void> {
  if (busy || !model) return
  busy = true
  while (pending) {
    const request = pending
    pending = null
    const start = performance.now()
    try {
      const prediction = await classify(model.session, ort.Tensor, model.meta, request.strokes)
      post({
        type: 'result',
        id: request.id,
        top: prediction.top,
        probs: prediction.probs,
        inputLength: prediction.inputLength,
        ms: performance.now() - start,
      })
    } catch (error) {
      post({ type: 'error', message: error instanceof Error ? error.message : String(error) })
    }
  }
  busy = false
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data
  if (message.type === 'init') {
    void init(message.modelsUrl)
  } else {
    pending = message
    void drain()
  }
}
