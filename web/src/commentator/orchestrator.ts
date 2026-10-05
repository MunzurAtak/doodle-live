/**
 * Commentary orchestrator (SPEC §8.3). Decides *when* to comment; the LLMClient decides
 * *what* to say.
 *
 * - At most one pending event (the latest wins) and one active generation.
 * - Cooldown: at least `cooldownMs` between the starts of two comments, except priority 3.
 * - A new event interrupts the active line if its priority is >= the active one's and the
 *   line has been visible for `minVisibleMs`; otherwise it waits as pending.
 * - Priority-3 events (round won / lost) always interrupt and clear the pending slot.
 * - Lines from previous rounds are dropped.
 */
import type { GameEvent, GameEventType } from './events'
import type { LLMClient } from './llm'
import { type ChatMessage, trimToSentence } from './prompt'

export type CommentSource = 'llm' | 'scripted'

export interface QueuedEvent {
  readonly event: GameEvent
  readonly round: number
  /** When the triggering stroke / state change happened (for latency metrics). */
  readonly triggeredAt: number
}

export interface CommentView {
  readonly id: number
  readonly round: number
  readonly event: GameEvent
  readonly text: string
  readonly streaming: boolean
  readonly source: CommentSource
}

export interface CommentMetric {
  readonly event: GameEventType
  readonly round: number
  readonly source: CommentSource
  /** Trigger to first visible token. null if no token arrived. */
  readonly firstTokenMs: number | null
  readonly totalMs: number
  readonly interrupted: boolean
  /** The LLM failed or returned nothing, so a scripted line was shown instead. */
  readonly fallback: boolean
  readonly text: string
}

export interface OrchestratorOptions {
  client: LLMClient
  source: CommentSource
  buildMessages(queued: QueuedEvent): readonly ChatMessage[]
  fallbackLine(event: GameEvent): string
  onView(view: CommentView): void
  onMetric?(metric: CommentMetric): void
  now?(): number
  cooldownMs?: number
  minVisibleMs?: number
}

interface Active {
  readonly id: number
  readonly queued: QueuedEvent
  readonly source: CommentSource
  readonly startedAt: number
  readonly controller: AbortController
  firstTokenAt: number | null
  text: string
  done: boolean
  interrupted: boolean
}

const RECHECK_MS = 100

export class CommentaryOrchestrator {
  private client: LLMClient
  private source: CommentSource
  private readonly opts: OrchestratorOptions
  private readonly now: () => number
  private readonly cooldownMs: number
  private readonly minVisibleMs: number

  private active: Active | null = null
  private pending: QueuedEvent | null = null
  private lastStartAt = -Infinity
  private timer: ReturnType<typeof setTimeout> | null = null
  private round = 0
  private nextId = 1
  private disposed = false

  constructor(opts: OrchestratorOptions) {
    this.opts = opts
    this.client = opts.client
    this.source = opts.source
    this.now = opts.now ?? (() => performance.now())
    this.cooldownMs = opts.cooldownMs ?? 1500
    this.minVisibleMs = opts.minVisibleMs ?? 800
  }

  /** Swap the model (e.g. once the LLM has loaded). Takes effect from the next comment. */
  setClient(client: LLMClient, source: CommentSource): void {
    this.client = client
    this.source = source
  }

  /** Call when a new round starts: drops pending and in-flight lines of older rounds. */
  setRound(round: number): void {
    if (round === this.round) return
    this.round = round
    if (this.pending && this.pending.round !== round) this.pending = null
    if (this.active && !this.active.done && this.active.queued.round !== round) {
      this.interrupt(this.active)
    }
  }

  push(event: GameEvent, round: number, triggeredAt: number = this.now()): void {
    if (this.disposed || round !== this.round) return
    this.pending = { event, round, triggeredAt }
    this.consider()
  }

  /** Stop everything (e.g. commentary switched off). */
  stop(): void {
    this.pending = null
    this.clearTimer()
    if (this.active && !this.active.done) this.interrupt(this.active)
  }

  dispose(): void {
    this.stop()
    this.disposed = true
  }

  private consider(): void {
    const queued = this.pending
    if (!queued || this.disposed) return
    if (queued.round !== this.round) {
      this.pending = null
      return
    }
    const now = this.now()
    const urgent = queued.event.priority === 3
    const running = this.active && !this.active.done ? this.active : null

    if (running && !urgent) {
      if (queued.event.priority < running.queued.event.priority) return // wait for it to finish
      if (running.firstTokenAt === null) return this.schedule(RECHECK_MS)
      const visibleFor = now - running.firstTokenAt
      if (visibleFor < this.minVisibleMs) return this.schedule(this.minVisibleMs - visibleFor)
    }
    if (!urgent && now - this.lastStartAt < this.cooldownMs) {
      return this.schedule(this.lastStartAt + this.cooldownMs - now)
    }

    this.pending = null
    this.clearTimer()
    if (running) this.interrupt(running)
    this.start(queued, now)
  }

  private start(queued: QueuedEvent, now: number): void {
    const active: Active = {
      id: this.nextId++,
      queued,
      source: this.source,
      startedAt: now,
      controller: new AbortController(),
      firstTokenAt: null,
      text: '',
      done: false,
      interrupted: false,
    }
    this.active = active
    this.lastStartAt = now
    this.emit(active, '', true) // replaces the previous line with a "thinking" state
    void this.run(active, this.client)
  }

  private async run(active: Active, client: LLMClient): Promise<void> {
    const { signal } = active.controller
    let fallback = false
    try {
      for await (const chunk of client.stream(this.opts.buildMessages(active.queued), signal)) {
        if (signal.aborted) break
        if (active.firstTokenAt === null) active.firstTokenAt = this.now()
        active.text += chunk
        this.emit(active, active.text, true)
      }
      if (!signal.aborted) {
        let text = trimToSentence(active.text)
        if (!text) {
          text = this.opts.fallbackLine(active.queued.event)
          fallback = true
        }
        active.text = text
        if (active.firstTokenAt === null) active.firstTokenAt = this.now()
        this.emit(active, text, false)
      }
    } catch {
      if (!signal.aborted) {
        fallback = true
        active.text = this.opts.fallbackLine(active.queued.event)
        if (active.firstTokenAt === null) active.firstTokenAt = this.now()
        this.emit(active, active.text, false)
      }
    } finally {
      active.done = true
      this.opts.onMetric?.({
        event: active.queued.event.type,
        round: active.queued.round,
        source: active.source,
        firstTokenMs:
          active.firstTokenAt === null ? null : active.firstTokenAt - active.queued.triggeredAt,
        totalMs: this.now() - active.startedAt,
        interrupted: active.interrupted,
        fallback,
        text: active.text,
      })
      this.consider()
    }
  }

  private interrupt(active: Active): void {
    active.interrupted = true
    active.controller.abort()
  }

  private emit(active: Active, text: string, streaming: boolean): void {
    if (this.disposed || active.interrupted) return
    this.opts.onView({
      id: active.id,
      round: active.queued.round,
      event: active.queued.event,
      text,
      streaming,
      source: active.source,
    })
  }

  private schedule(ms: number): void {
    this.clearTimer()
    this.timer = setTimeout(
      () => {
        this.timer = null
        this.consider()
      },
      Math.max(0, ms),
    )
  }

  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
  }
}
