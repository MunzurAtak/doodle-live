/**
 * Stroke preprocessing (SPEC §4). Must stay identical to `ml/src/doodle_ml/preprocess.py`.
 *
 * 1. normalise  shift to the origin, scale uniformly so the longer side is 255
 * 2. resample   insert points so consecutive points are at most `resampleSpacing` apart
 * 3. simplify   Ramer-Douglas-Peucker with `rdpEpsilon`
 * 4. stroke-3   rows of (dx, dy, penLift); the first offset is from (0, 0)
 * 5. scale      dx, dy divided by `offsetScale`
 * 6. pad        truncate / zero-pad to `maxLen` rows, plus a 0/1 mask
 *
 * Every arithmetic expression mirrors the Python code in the same order, so both
 * produce the same float64 values. The shared golden fixtures in
 * `shared/fixtures/preprocess_cases.json` enforce this.
 */

export interface Point {
  readonly x: number
  readonly y: number
}

export type Stroke = readonly Point[]
export type Drawing = readonly Stroke[]

export interface PreprocessConfig {
  readonly maxLen: number
  readonly rdpEpsilon: number
  readonly resampleSpacing: number
  readonly offsetScale: number
}

export interface PreprocessResult {
  /** Row-major `maxLen x 3` stroke-3 rows (dx, dy, penLift); padding rows are zero. */
  readonly strokes: Float32Array
  /** `maxLen` values: 1 for real rows, 0 for padding. */
  readonly mask: Float32Array
  /** Number of real rows (after truncation). */
  readonly length: number
}

export const DEFAULT_PREPROCESS_CONFIG: PreprocessConfig = {
  maxLen: 200,
  rdpEpsilon: 2.0,
  resampleSpacing: 1.0,
  offsetScale: 1.0,
}

const TARGET_SIZE = 255.0

/** Step 1: shift to the origin and scale the longer side to 255. Drops empty strokes. */
export function normalize(drawing: Drawing): Point[][] {
  const strokes = drawing.filter((s) => s.length > 0)
  if (strokes.length === 0) return []
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const stroke of strokes) {
    for (const p of stroke) {
      if (p.x < minX) minX = p.x
      if (p.y < minY) minY = p.y
      if (p.x > maxX) maxX = p.x
      if (p.y > maxY) maxY = p.y
    }
  }
  const extent = Math.max(maxX - minX, maxY - minY)
  if (extent === 0) {
    return strokes.map((s) => s.map((p) => ({ x: p.x - minX, y: p.y - minY })))
  }
  // Multiply before dividing so the longest side maps to exactly 255.0.
  return strokes.map((s) =>
    s.map((p) => ({
      x: ((p.x - minX) * TARGET_SIZE) / extent,
      y: ((p.y - minY) * TARGET_SIZE) / extent,
    })),
  )
}

/** Step 2: linear interpolation so consecutive points are at most `spacing` apart. */
export function resampleStroke(stroke: Stroke, spacing: number): Point[] {
  if (stroke.length === 0) return []
  const out: Point[] = [stroke[0]]
  for (let i = 1; i < stroke.length; i++) {
    const { x: x0, y: y0 } = out[out.length - 1]
    const { x: x1, y: y1 } = stroke[i]
    const dx = x1 - x0
    const dy = y1 - y0
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (dist === 0) continue
    const n = Math.ceil(dist / spacing)
    for (let k = 1; k < n; k++) {
      const t = k / n
      out.push({ x: x0 + dx * t, y: y0 + dy * t })
    }
    out.push({ x: x1, y: y1 })
  }
  return out
}

/** Distance from `p` to the segment `a`-`b`. */
function segmentDistance(p: Point, a: Point, b: Point): number {
  const vx = b.x - a.x
  const vy = b.y - a.y
  const wx = p.x - a.x
  const wy = p.y - a.y
  const lengthSq = vx * vx + vy * vy
  if (lengthSq === 0) return Math.sqrt(wx * wx + wy * wy)
  let t = (wx * vx + wy * vy) / lengthSq
  if (t < 0) t = 0
  else if (t > 1) t = 1
  const cx = a.x + vx * t - p.x
  const cy = a.y + vy * t - p.y
  return Math.sqrt(cx * cx + cy * cy)
}

/** Step 3: Ramer-Douglas-Peucker simplification (iterative, keeps endpoints). */
export function rdp(stroke: Stroke, epsilon: number): Point[] {
  const n = stroke.length
  if (n <= 2) return [...stroke]
  const keep = new Array<boolean>(n).fill(false)
  keep[0] = true
  keep[n - 1] = true
  const stack: Array<[number, number]> = [[0, n - 1]]
  while (stack.length > 0) {
    const [start, end] = stack.pop() as [number, number]
    let maxDist = -1
    let index = -1
    for (let i = start + 1; i < end; i++) {
      const d = segmentDistance(stroke[i], stroke[start], stroke[end])
      if (d > maxDist) {
        maxDist = d
        index = i
      }
    }
    if (index !== -1 && maxDist > epsilon) {
      keep[index] = true
      stack.push([start, index])
      stack.push([index, end])
    }
  }
  return stroke.filter((_, i) => keep[i])
}

/** Steps 1-3: normalise, resample, simplify. Returns absolute coordinates. */
export function simplifyDrawing(
  drawing: Drawing,
  config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
): Point[][] {
  return normalize(drawing).map((s) =>
    rdp(resampleStroke(s, config.resampleSpacing), config.rdpEpsilon),
  )
}

/** Step 4: (dx, dy, penLift) rows; penLift is 1 on the last point of each stroke. */
export function toStroke3(drawing: Drawing): Array<[number, number, number]> {
  const rows: Array<[number, number, number]> = []
  let prevX = 0
  let prevY = 0
  for (const stroke of drawing) {
    const last = stroke.length - 1
    stroke.forEach((p, i) => {
      rows.push([p.x - prevX, p.y - prevY, i === last ? 1 : 0])
      prevX = p.x
      prevY = p.y
    })
  }
  return rows
}

/** Steps 4-6 on an already simplified drawing. */
export function encode(
  drawing: Drawing,
  config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
): PreprocessResult {
  const rows = toStroke3(drawing).slice(0, config.maxLen)
  const strokes = new Float32Array(config.maxLen * 3)
  const mask = new Float32Array(config.maxLen)
  rows.forEach(([dx, dy, lift], i) => {
    strokes[i * 3] = dx / config.offsetScale
    strokes[i * 3 + 1] = dy / config.offsetScale
    strokes[i * 3 + 2] = lift
    mask[i] = 1
  })
  return { strokes, mask, length: rows.length }
}

/** Full pipeline (steps 1-6), as used on the live canvas. */
export function preprocess(
  drawing: Drawing,
  config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
): PreprocessResult {
  return encode(simplifyDrawing(drawing, config), config)
}
