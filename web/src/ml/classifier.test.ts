import { describe, expect, it, vi } from 'vitest'
import { ClassifierClient, type WorkerLike } from './classifier'
import type { WorkerRequest, WorkerResponse } from './protocol'

class FakeWorker implements WorkerLike {
  sent: WorkerRequest[] = []
  terminated = false
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null
  postMessage(message: WorkerRequest) {
    this.sent.push(message)
  }
  terminate() {
    this.terminated = true
  }
  reply(message: WorkerResponse) {
    this.onmessage?.({ data: message } as MessageEvent<WorkerResponse>)
  }
}

const result = (id: number): WorkerResponse => ({
  type: 'result',
  id,
  top: [],
  probs: [],
  inputLength: 1,
  ms: 1,
})

function setup() {
  const worker = new FakeWorker()
  const callbacks = { onReady: vi.fn(), onResult: vi.fn(), onError: vi.fn() }
  const client = new ClassifierClient(worker, 'https://x/models/', callbacks)
  return { worker, callbacks, client }
}

describe('ClassifierClient', () => {
  it('initialises the worker with the models URL', () => {
    const { worker, callbacks } = setup()
    expect(worker.sent[0]).toEqual({ type: 'init', modelsUrl: 'https://x/models/' })
    worker.reply({ type: 'ready', categories: ['cat'] })
    expect(callbacks.onReady).toHaveBeenCalledWith(['cat'])
  })

  it('sends increasing ids and drops results older than the newest delivered', () => {
    const { worker, callbacks, client } = setup()
    expect(client.classify([[{ x: 0, y: 0 }]])).toBe(1)
    expect(client.classify([[{ x: 1, y: 1 }]])).toBe(2)
    expect(worker.sent.slice(1).map((m) => m.type)).toEqual(['classify', 'classify'])

    worker.reply(result(2))
    worker.reply(result(1)) // stale: arrives after a newer one
    expect(callbacks.onResult).toHaveBeenCalledTimes(1)
    expect(callbacks.onResult.mock.calls[0][0].id).toBe(2)
  })

  it('reports errors and terminates the worker on dispose', () => {
    const { worker, callbacks, client } = setup()
    worker.reply({ type: 'error', message: 'boom' })
    expect(callbacks.onError).toHaveBeenCalledWith('boom')
    client.dispose()
    expect(worker.terminated).toBe(true)
  })
})
