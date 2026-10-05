import type { InferenceSession, Tensor } from 'onnxruntime-web'
import { type ModelMeta, preprocessConfigFromMeta } from './meta'
import { type Drawing, preprocess } from './preprocess'

export interface Guess {
  readonly label: string
  readonly prob: number
}

export interface Prediction {
  /** Top guesses, most likely first. Empty for an empty drawing. */
  readonly top: readonly Guess[]
  /** Probability per category, in `meta.categories` order. */
  readonly probs: readonly number[]
  readonly logits: readonly number[]
  /** Number of stroke-3 rows the model saw (after preprocessing and truncation). */
  readonly inputLength: number
}

/** The Tensor constructor of whichever onnxruntime package created the session. */
export type TensorConstructor = new (
  type: 'float32',
  data: Float32Array,
  dims: readonly number[],
) => Tensor

export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i])
  const exps = Array.from(logits, (v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

export function topK(probs: readonly number[], labels: readonly string[], k: number): Guess[] {
  return probs
    .map((prob, i) => ({ label: labels[i], prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k)
}

export function emptyPrediction(meta: ModelMeta): Prediction {
  const zeros = new Array<number>(meta.categories.length).fill(0)
  return { top: [], probs: zeros, logits: zeros, inputLength: 0 }
}

/** Preprocess a drawing and run the classifier on it. */
export async function classify(
  session: InferenceSession,
  TensorClass: TensorConstructor,
  meta: ModelMeta,
  drawing: Drawing,
  k = 5,
): Promise<Prediction> {
  const input = preprocess(drawing, preprocessConfigFromMeta(meta))
  if (input.length === 0) return emptyPrediction(meta)
  const [strokesName, maskName] = meta.inputNames
  const outputs = await session.run({
    [strokesName]: new TensorClass('float32', input.strokes, [1, meta.maxLen, 3]),
    [maskName]: new TensorClass('float32', input.mask, [1, meta.maxLen]),
  })
  const logits = Array.from(outputs[meta.outputName].data as Float32Array)
  const probs = softmax(logits)
  return { top: topK(probs, meta.categories, k), probs, logits, inputLength: input.length }
}
