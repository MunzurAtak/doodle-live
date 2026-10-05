import type { InferenceSession } from 'onnxruntime-web'
import { describe, expect, it, vi } from 'vitest'
import { type TensorConstructor, classify, softmax, topK } from './inference'
import type { ModelMeta } from './meta'

const META: ModelMeta = {
  version: 1,
  categories: ['cat', 'dog', 'sun'],
  maxLen: 8,
  offsetScale: 10,
  rdpEpsilon: 2,
  resampleSpacing: 1,
  inputNames: ['strokes', 'mask'],
  outputName: 'logits',
}

class FakeTensor {
  readonly type: string
  readonly data: Float32Array
  readonly dims: readonly number[]
  constructor(type: string, data: Float32Array, dims: readonly number[]) {
    this.type = type
    this.data = data
    this.dims = dims
  }
}

describe('softmax / topK', () => {
  it('softmax sums to 1, keeps order and is numerically stable', () => {
    const probs = softmax([1000, 1001, 999])
    expect(probs.reduce((a, b) => a + b)).toBeCloseTo(1, 10)
    expect(probs[1]).toBeGreaterThan(probs[0])
    expect(probs.every(Number.isFinite)).toBe(true)
  })

  it('topK returns labels sorted by probability', () => {
    expect(topK([0.2, 0.5, 0.3], ['a', 'b', 'c'], 2)).toEqual([
      { label: 'b', prob: 0.5 },
      { label: 'c', prob: 0.3 },
    ])
  })
})

describe('classify', () => {
  it('feeds preprocessed tensors with names and shapes from the meta', async () => {
    const run = vi.fn(async (feeds: Record<string, FakeTensor>) => {
      expect(feeds.strokes.dims).toEqual([1, 8, 3])
      expect(feeds.mask.dims).toEqual([1, 8])
      expect(Array.from(feeds.mask.data)).toEqual([1, 1, 0, 0, 0, 0, 0, 0])
      return { logits: { data: new Float32Array([0, 2, 1]) } }
    })
    const session = { run } as unknown as InferenceSession
    const prediction = await classify(session, FakeTensor as unknown as TensorConstructor, META, [
      [
        { x: 0, y: 0 },
        { x: 50, y: 0 },
      ],
    ])
    expect(run).toHaveBeenCalledOnce()
    expect(prediction.top.map((g) => g.label)).toEqual(['dog', 'sun', 'cat'])
    expect(prediction.inputLength).toBe(2)
  })

  it('does not run the model for an empty drawing', async () => {
    const run = vi.fn()
    const prediction = await classify(
      { run } as unknown as InferenceSession,
      FakeTensor as unknown as TensorConstructor,
      META,
      [],
    )
    expect(run).not.toHaveBeenCalled()
    expect(prediction.top).toEqual([])
  })
})
