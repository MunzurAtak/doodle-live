import { useEffect, useRef, useState } from 'react'
import type { GameEvent } from './events'
import { scriptedLine } from './fallbackLines'
import { type LLMClient, ScriptedClient } from './llm'
import {
  type CommentMetric,
  type CommentView,
  CommentaryOrchestrator,
  type QueuedEvent,
} from './orchestrator'
import { type Personality } from './personalities'
import { type PromptContext, buildMessages } from './prompt'
import { createSpeaker } from './tts'

interface Options {
  /** Latest event and a counter that increases with every new event. */
  readonly event: GameEvent | null
  readonly eventSeq: number
  readonly round: number
  readonly enabled: boolean
  readonly voice: boolean
  readonly personality: Personality
  /** The loaded LLM, or null to use scripted lines. */
  readonly llm: LLMClient | null
  readonly context: Omit<PromptContext, 'previousLines'>
}

const MAX_METRICS = 500

/** Connects game events to the orchestrator and returns the line to show. */
export function useCommentator({
  event,
  eventSeq,
  round,
  enabled,
  voice,
  personality,
  llm,
  context,
}: Options) {
  const [view, setView] = useState<CommentView | null>(null)
  const [metrics, setMetrics] = useState<readonly CommentMetric[]>([])
  const orchestrator = useRef<CommentaryOrchestrator | null>(null)
  const scripted = useRef(new ScriptedClient())
  const contextRef = useRef(context)
  const personalityRef = useRef(personality)
  const lines = useRef<{ round: number; text: string }[]>([])
  const speaker = useRef(createSpeaker())
  const voiceRef = useRef(voice)

  useEffect(() => {
    contextRef.current = context
    personalityRef.current = personality
    voiceRef.current = voice
    if (!voice) speaker.current.cancel()
  })

  useEffect(() => {
    const o = new CommentaryOrchestrator({
      client: scripted.current,
      source: 'scripted',
      buildMessages: (queued: QueuedEvent) =>
        buildMessages(
          queued.event,
          {
            ...contextRef.current,
            previousLines: lines.current.filter((l) => l.round === queued.round).map((l) => l.text),
          },
          personalityRef.current,
        ),
      fallbackLine: (e) => scriptedLine(e),
      onView: (v) => {
        setView(v)
        if (voiceRef.current) speaker.current.update(v.id, v.text, !v.streaming)
        if (!v.streaming)
          lines.current = [...lines.current.slice(-20), { round: v.round, text: v.text }]
      },
      onMetric: (m) => setMetrics((all) => [...all.slice(-(MAX_METRICS - 1)), m]),
    })
    orchestrator.current = o
    const voiceOut = speaker.current
    return () => {
      o.dispose()
      voiceOut.cancel()
    }
  }, [])

  useEffect(() => {
    orchestrator.current?.setClient(llm ?? scripted.current, llm ? 'llm' : 'scripted')
  }, [llm])

  useEffect(() => {
    orchestrator.current?.setRound(round)
  }, [round])

  useEffect(() => {
    if (!enabled) {
      orchestrator.current?.stop()
      speaker.current.cancel()
    }
  }, [enabled])

  useEffect(() => {
    if (event && enabled) orchestrator.current?.push(event, round)
    // A new eventSeq is the trigger; `event` changes with it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eventSeq])

  const visible = enabled && view && view.round === round ? view : null
  return { view: visible, metrics }
}
