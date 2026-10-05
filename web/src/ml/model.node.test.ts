// @vitest-environment node
/**
 * Runs the real exported model with onnxruntime-web (the same runtime as the browser,
 * here on Node's WebAssembly) and checks it against the outputs recorded by the Python
 * export step. Catches preprocessing drift, wrong tensor names and quantization issues.
 */
import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web'
import { beforeAll, describe, expect, it } from 'vitest'
import cases from '../../../shared/fixtures/model_cases.json'
import rawMeta from '../../public/models/model_meta.json'
import { classify } from './inference'
import { parseModelMeta } from './meta'

const meta = parseModelMeta(rawMeta)
let session: ort.InferenceSession

beforeAll(async () => {
  ort.env.wasm.numThreads = 1
  const model = readFileSync(new URL('../../public/models/classifier.onnx', import.meta.url))
  session = await ort.InferenceSession.create(model)
})

describe('exported classifier (onnxruntime-web)', () => {
  it('has the inputs and output named in model_meta.json', () => {
    expect(session.inputNames).toEqual([...meta.inputNames])
    expect(session.outputNames).toEqual([meta.outputName])
  })

  it.each(cases.cases.map((c) => [c.label, c] as const))(
    'matches the Python export for a "%s" drawing',
    async (_label, c) => {
      const drawing = c.input.map((stroke) => stroke.map(([x, y]) => ({ x, y })))
      const prediction = await classify(session, ort.Tensor, meta, drawing, 3)
      expect(prediction.top.map((g) => g.label)).toEqual(c.expected_top3)
      prediction.logits.forEach((value, i) =>
        expect(Math.abs(value - c.expected_logits[i])).toBeLessThanOrEqual(cases.logit_tolerance),
      )
    },
  )
})
