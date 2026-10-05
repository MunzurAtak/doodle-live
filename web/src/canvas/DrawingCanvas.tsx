import { type PointerEvent, useEffect, useRef } from 'react'
import type { Point } from '../ml/preprocess'
import { type StrokesAction, type StrokesState, visibleStrokes } from './strokes'

/** Logical drawing space. Pointer positions are mapped into it, so stroke coordinates
 * don't depend on the on-screen canvas size (and preprocessing rescales anyway). */
export const LOGICAL_SIZE = 512

interface DrawingCanvasProps {
  readonly state: StrokesState
  readonly dispatch: (action: StrokesAction) => void
  readonly disabled?: boolean
  readonly strokeColor?: string
}

export function DrawingCanvas({
  state,
  dispatch,
  disabled = false,
  strokeColor = '#1e293b',
}: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const draw = () => render(canvas, visibleStrokes(state), strokeColor)
    draw()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [state, strokeColor])

  const toPoint = (event: PointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) * LOGICAL_SIZE) / rect.width,
      y: ((event.clientY - rect.top) * LOGICAL_SIZE) / rect.height,
    }
  }

  return (
    <canvas
      ref={canvasRef}
      aria-label="Drawing canvas"
      className="aspect-square w-full touch-none rounded-2xl bg-white shadow-inner select-none"
      style={{ touchAction: 'none', cursor: disabled ? 'not-allowed' : 'crosshair' }}
      onPointerDown={(event) => {
        if (disabled || (event.pointerType === 'mouse' && event.button !== 0)) return
        event.currentTarget.setPointerCapture(event.pointerId)
        dispatch({ type: 'start', point: toPoint(event) })
      }}
      onPointerMove={(event) => {
        if (!disabled && event.currentTarget.hasPointerCapture(event.pointerId)) {
          dispatch({ type: 'move', point: toPoint(event) })
        }
      }}
      onPointerUp={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId)
        }
        dispatch({ type: 'end' })
      }}
      onPointerCancel={() => dispatch({ type: 'end' })}
    />
  )
}

function render(
  canvas: HTMLCanvasElement,
  strokes: readonly (readonly Point[])[],
  color: string,
): void {
  const dpr = window.devicePixelRatio || 1
  const width = Math.round(canvas.clientWidth * dpr)
  const height = Math.round(canvas.clientHeight * dpr)
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(width / LOGICAL_SIZE, 0, 0, height / LOGICAL_SIZE, 0, 0)
  ctx.clearRect(0, 0, LOGICAL_SIZE, LOGICAL_SIZE)
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = 6
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  for (const stroke of strokes) {
    if (stroke.length === 1) {
      ctx.beginPath()
      ctx.arc(stroke[0].x, stroke[0].y, ctx.lineWidth / 2, 0, Math.PI * 2)
      ctx.fill()
      continue
    }
    ctx.beginPath()
    stroke.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)))
    ctx.stroke()
  }
}
