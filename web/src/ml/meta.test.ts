import { describe, expect, it } from 'vitest'
import rawMeta from '../../public/models/model_meta.json'
import { modelsBaseUrl, parseModelMeta, preprocessConfigFromMeta } from './meta'

describe('model meta', () => {
  it('parses the exported model_meta.json', () => {
    const meta = parseModelMeta(rawMeta)
    expect(meta.categories).toHaveLength(50)
    expect(meta.maxLen).toBe(200)
    expect(meta.inputNames).toEqual(['strokes', 'mask'])
    expect(preprocessConfigFromMeta(meta)).toEqual({
      maxLen: 200,
      rdpEpsilon: 2,
      resampleSpacing: 1,
      offsetScale: rawMeta.offset_scale,
    })
  })

  it.each([
    ['not an object', []],
    ['missing categories', { ...rawMeta, categories: [] }],
    ['bad offset_scale', { ...rawMeta, offset_scale: 0 }],
    ['bad input_names', { ...rawMeta, input_names: ['strokes'] }],
    ['bad output_name', { ...rawMeta, output_name: 3 }],
  ])('rejects %s', (_name, raw) => {
    expect(() => parseModelMeta(raw)).toThrow(/model_meta\.json/)
  })

  it('builds the models URL under the GitHub Pages base path', () => {
    expect(modelsBaseUrl('https://x.github.io/doodle-live/')).toBe(
      'https://x.github.io/doodle-live/models/',
    )
  })
})
