import { describe, expect, it, vi } from 'vitest'
import { makeEvent } from './events'
import { type ChatEngine, ScriptedClient, WebLLMClient, eventFromMessages } from './llm'
import { chooseModelId, detectWebGPU } from './models'
import { buildMessages } from './prompt'
import { getPersonality } from './personalities'

const collect = async (it: AsyncIterable<string>) => {
  const out: string[] = []
  for await (const chunk of it) out.push(chunk)
  return out
}

const messagesFor = (type: Parameters<typeof makeEvent>[0], answer: string | null = null) =>
  buildMessages(
    makeEvent(type),
    {
      strokes: 2,
      secondsLeft: 9,
      candidates: [{ label: 'cat', prob: 0.4 }],
      previousLines: ['Hello.'],
      answer,
    },
    getPersonality('host'),
  )

function fakeEngine(chunks: string[]) {
  const interruptGenerate = vi.fn()
  const create = vi.fn(async () =>
    (async function* () {
      for (const c of chunks) yield { choices: [{ delta: { content: c } }] }
    })(),
  )
  return {
    engine: { chat: { completions: { create } }, interruptGenerate } as ChatEngine,
    create,
    interruptGenerate,
  }
}

describe('WebLLMClient', () => {
  it('streams delta content with the spec generation settings', async () => {
    const { engine, create } = fakeEngine(['Is ', 'that ', 'a cat?'])
    const out = await collect(
      new WebLLMClient(engine).stream(messagesFor('FIRST_GUESS'), new AbortController().signal),
    )
    expect(out.join('')).toBe('Is that a cat?')
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ stream: true, temperature: 0.8, max_tokens: 48 }),
    )
  })

  it('interrupts the engine when aborted', async () => {
    const { engine, interruptGenerate } = fakeEngine(['a', 'b', 'c'])
    const controller = new AbortController()
    const out: string[] = []
    for await (const chunk of new WebLLMClient(engine).stream(
      messagesFor('FIRST_GUESS'),
      controller.signal,
    )) {
      out.push(chunk)
      controller.abort()
    }
    expect(out).toEqual(['a'])
    expect(interruptGenerate).toHaveBeenCalled()
  })
})

describe('ScriptedClient', () => {
  it('reads the event from the prompt and returns one scripted line', async () => {
    const out = await collect(
      new ScriptedClient(() => 0).stream(
        messagesFor('ROUND_WON', 'cat'),
        new AbortController().signal,
      ),
    )
    expect(out).toHaveLength(1)
    expect(out[0].toLowerCase()).toContain('cat')
    expect(eventFromMessages(messagesFor('ROUND_WON', 'cat')).event).toMatchObject({
      type: 'ROUND_WON',
      answer: 'cat',
    })
  })
})

describe('models', () => {
  it('picks the f16 build only when the GPU supports it', () => {
    expect(chooseModelId('default', true)).toBe('Qwen2.5-1.5B-Instruct-q4f16_1-MLC')
    expect(chooseModelId('light', false)).toBe('Qwen2.5-0.5B-Instruct-q4f32_1-MLC')
  })

  it('detects missing or unusable WebGPU', async () => {
    expect((await detectWebGPU(undefined)).ok).toBe(false)
    expect((await detectWebGPU({ requestAdapter: async () => null })).ok).toBe(false)
    expect(
      await detectWebGPU({ requestAdapter: async () => ({ features: new Set(['shader-f16']) }) }),
    ).toEqual({ ok: true, shaderF16: true })
  })
})
