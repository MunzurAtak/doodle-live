import { describe, expect, it } from 'vitest'
import fixture from '../../../shared/fixtures/preprocess_cases.json'
import {
  type Drawing,
  type PreprocessConfig,
  normalize,
  preprocess,
  rdp,
  resampleStroke,
  simplifyDrawing,
  toStroke3,
} from './preprocess'

interface FixtureCase {
  name: string
  config: { max_len: number; rdp_epsilon: number; resample_spacing: number; offset_scale: number }
  input: number[][][]
  expected: { simplified: number[][][]; length: number; rows: number[][] }
}

const cases = fixture.cases as FixtureCase[]
const TOLERANCE = fixture.tolerance

const toDrawing = (raw: number[][][]): Drawing =>
  raw.map((stroke) => stroke.map(([x, y]) => ({ x, y })))

const toConfig = (c: FixtureCase['config']): PreprocessConfig => ({
  maxLen: c.max_len,
  rdpEpsilon: c.rdp_epsilon,
  resampleSpacing: c.resample_spacing,
  offsetScale: c.offset_scale,
})

const pts = (...xy: Array<[number, number]>) => xy.map(([x, y]) => ({ x, y }))

describe('preprocess: golden fixtures shared with Python', () => {
  it('has at least 20 cases', () => {
    expect(cases.length).toBeGreaterThanOrEqual(20)
  })

  it.each(cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const config = toConfig(c.config)
    const drawing = toDrawing(c.input)

    const simplified = simplifyDrawing(drawing, config)
    expect(simplified.map((s) => s.length)).toEqual(c.expected.simplified.map((s) => s.length))
    simplified.forEach((stroke, i) =>
      stroke.forEach((p, j) => {
        const [ex, ey] = c.expected.simplified[i][j]
        expect(Math.abs(p.x - ex)).toBeLessThanOrEqual(TOLERANCE)
        expect(Math.abs(p.y - ey)).toBeLessThanOrEqual(TOLERANCE)
      }),
    )

    const result = preprocess(drawing, config)
    expect(result.length).toBe(c.expected.length)
    expect(result.strokes.length).toBe(config.maxLen * 3)
    expect(result.mask.length).toBe(config.maxLen)
    for (let i = 0; i < config.maxLen; i++) {
      const real = i < c.expected.length
      expect(result.mask[i]).toBe(real ? 1 : 0)
      for (let k = 0; k < 3; k++) {
        const expected = real ? c.expected.rows[i][k] : 0
        expect(Math.abs(result.strokes[i * 3 + k] - expected)).toBeLessThanOrEqual(TOLERANCE)
      }
    }
  })
})

describe('preprocess: unit behaviour', () => {
  it('normalises to the origin and scales the longest side to exactly 255', () => {
    expect(normalize([pts([10, 20], [60, 30])])).toEqual([pts([0, 0], [255, 51])])
  })

  it('drops empty strokes and handles an empty drawing', () => {
    expect(normalize([[], pts([1, 1])])).toEqual([pts([0, 0])])
    const result = preprocess([])
    expect(result.length).toBe(0)
    expect(result.strokes.every((v) => v === 0)).toBe(true)
  })

  it('resamples so gaps are at most the spacing', () => {
    expect(resampleStroke(pts([0, 0], [3.5, 0]), 1)).toEqual(
      pts([0, 0], [0.875, 0], [1.75, 0], [2.625, 0], [3.5, 0]),
    )
  })

  it('rdp removes collinear points and uses a strict threshold', () => {
    expect(rdp(pts([0, 0], [5, 2], [10, 0]), 2)).toEqual(pts([0, 0], [10, 0]))
    expect(rdp(pts([0, 0], [5, 2.001], [10, 0]), 2)).toHaveLength(3)
  })

  it('builds stroke-3 rows with pen lifts', () => {
    expect(toStroke3([pts([1, 2], [4, 6]), pts([10, 10])])).toEqual([
      [1, 2, 0],
      [3, 4, 1],
      [6, 4, 1],
    ])
  })
})
