import type { Point } from './preprocess'
import type { WorkerRequest, WorkerResponse } from './protocol'

export type ClassifierResult = Extract<WorkerResponse, { type: 'result' }>

export interface WorkerLike {
  postMessage(message: WorkerRequest): void
  terminate(): void
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null
}

export interface ClassifierCallbacks {
  onReady(categories: readonly string[]): void
  onResult(result: ClassifierResult): void
  onError(message: string): void
}

export function createClassifierWorker(): WorkerLike {
  return new Worker(new URL('./classifier.worker.ts', import.meta.url), {
    type: 'module',
  }) as WorkerLike
}

/**
 * Main-thread client for the classifier worker. Every request gets an increasing id and
 * results older than the newest one already delivered are dropped, so the UI never
 * jumps back to a stale prediction.
 */
export class ClassifierClient {
  private nextId = 0
  private lastDelivered = 0
  private readonly worker: WorkerLike
  private readonly callbacks: ClassifierCallbacks

  constructor(worker: WorkerLike, modelsUrl: string, callbacks: ClassifierCallbacks) {
    this.worker = worker
    this.callbacks = callbacks
    worker.onmessage = (event) => this.handle(event.data)
    worker.postMessage({ type: 'init', modelsUrl })
  }

  classify(strokes: readonly (readonly Point[])[]): number {
    const id = ++this.nextId
    this.worker.postMessage({ type: 'classify', id, strokes })
    return id
  }

  dispose(): void {
    this.worker.onmessage = null
    this.worker.terminate()
  }

  private handle(message: WorkerResponse): void {
    switch (message.type) {
      case 'ready':
        this.callbacks.onReady(message.categories)
        break
      case 'error':
        this.callbacks.onError(message.message)
        break
      case 'result':
        if (message.id <= this.lastDelivered) return
        this.lastDelivered = message.id
        this.callbacks.onResult(message)
        break
    }
  }
}
