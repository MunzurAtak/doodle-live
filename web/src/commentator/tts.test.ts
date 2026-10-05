import { describe, expect, it, vi } from 'vitest'
import { completedSentences, createSpeaker } from './tts'

describe('completedSentences', () => {
  it('returns only finished sentences and where to continue', () => {
    expect(completedSentences('Is that a cat? No, a do', 0)).toEqual({
      sentences: ['Is that a cat?'],
      next: 15,
    })
    expect(completedSentences('Is that a cat? No, a dog!', 15)).toEqual({
      sentences: ['No, a dog!'],
      next: 25,
    })
  })
})

describe('createSpeaker', () => {
  it('speaks sentences once, the rest at the end, and cancels on a new comment', () => {
    const spoken: string[] = []
    const synth = { speak: vi.fn((u: { text: string }) => spoken.push(u.text)), cancel: vi.fn() }
    vi.stubGlobal(
      'SpeechSynthesisUtterance',
      class {
        text: string
        rate = 1
        constructor(t: string) {
          this.text = t
        }
      },
    )
    const speaker = createSpeaker(synth as unknown as SpeechSynthesis)
    speaker.update(1, 'Hmm. Is it', false)
    speaker.update(1, 'Hmm. Is it a cat', false)
    speaker.update(1, 'Hmm. Is it a cat', true)
    expect(spoken).toEqual(['Hmm.', 'Is it a cat'])
    speaker.update(2, 'Dog!', true)
    expect(synth.cancel).toHaveBeenCalled()
    expect(spoken.at(-1)).toBe('Dog!')
    vi.unstubAllGlobals()
  })
})
