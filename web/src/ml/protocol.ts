import type { Guess } from './inference'
import type { Point } from './preprocess'

/** Messages between the main thread and `classifier.worker.ts`. */
export type WorkerRequest =
  | { readonly type: 'init'; readonly modelsUrl: string }
  | { readonly type: 'classify'; readonly id: number; readonly strokes: readonly (readonly Point[])[] }

export type WorkerResponse =
  | { readonly type: 'ready'; readonly categories: readonly string[] }
  | { readonly type: 'error'; readonly message: string }
  | {
      readonly type: 'result'
      readonly id: number
      readonly top: readonly Guess[]
      readonly probs: readonly number[]
      readonly inputLength: number
      readonly ms: number
    }
