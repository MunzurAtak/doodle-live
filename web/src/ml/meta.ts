import type { PreprocessConfig } from './preprocess'

/**
 * `public/models/model_meta.json` (written by `ml/src/doodle_ml/export.py`) is the single
 * source of truth for labels, preprocessing parameters and tensor names on the web side.
 */
export interface ModelMeta {
  readonly version: number
  readonly categories: readonly string[]
  readonly maxLen: number
  readonly offsetScale: number
  readonly rdpEpsilon: number
  readonly resampleSpacing: number
  readonly inputNames: readonly [strokes: string, mask: string]
  readonly outputName: string
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

function positiveNumber(raw: Record<string, unknown>, key: string): number {
  const value = raw[key]
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`model_meta.json: "${key}" must be a positive number`)
  }
  return value
}

/** Validate the raw JSON and convert it to camelCase. Throws on anything unexpected. */
export function parseModelMeta(raw: unknown): ModelMeta {
  if (!isRecord(raw)) throw new Error('model_meta.json: expected an object')
  const { categories, input_names: inputNames, output_name: outputName } = raw
  if (
    !Array.isArray(categories) ||
    categories.length === 0 ||
    !categories.every((c) => typeof c === 'string')
  ) {
    throw new Error('model_meta.json: "categories" must be a non-empty string array')
  }
  if (
    !Array.isArray(inputNames) ||
    inputNames.length !== 2 ||
    !inputNames.every((n) => typeof n === 'string')
  ) {
    throw new Error('model_meta.json: "input_names" must be [strokes, mask]')
  }
  if (typeof outputName !== 'string') {
    throw new Error('model_meta.json: "output_name" must be a string')
  }
  return {
    version: positiveNumber(raw, 'version'),
    categories: categories as string[],
    maxLen: positiveNumber(raw, 'max_len'),
    offsetScale: positiveNumber(raw, 'offset_scale'),
    rdpEpsilon: positiveNumber(raw, 'rdp_epsilon'),
    resampleSpacing: positiveNumber(raw, 'resample_spacing'),
    inputNames: [inputNames[0] as string, inputNames[1] as string],
    outputName,
  }
}

export function preprocessConfigFromMeta(meta: ModelMeta): PreprocessConfig {
  return {
    maxLen: meta.maxLen,
    rdpEpsilon: meta.rdpEpsilon,
    resampleSpacing: meta.resampleSpacing,
    offsetScale: meta.offsetScale,
  }
}

/** Absolute URL of the folder holding classifier.onnx and model_meta.json. */
export function modelsBaseUrl(appBaseUrl: string): string {
  return new URL('models/', appBaseUrl).href
}
