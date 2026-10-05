/** Optional voice (SPEC §8.6): speak each sentence as soon as it is complete. */

/** Complete sentences in `text` after character `from`. */
export function completedSentences(
  text: string,
  from: number,
): { sentences: string[]; next: number } {
  const sentences: string[] = []
  const pattern = /[^.!?]*[.!?]+["')\]]?(\s|$)/g
  pattern.lastIndex = from
  let next = from
  let match: RegExpExecArray | null
  while ((match = pattern.exec(text)) !== null && match.index === next) {
    const sentence = match[0].trim()
    if (sentence) sentences.push(sentence)
    next = pattern.lastIndex
  }
  return { sentences, next }
}

export interface Speaker {
  /** Feed the text of the current line so far; speaks newly completed sentences. */
  update(commentId: number, text: string, final: boolean): void
  cancel(): void
}

export function createSpeaker(
  synth: SpeechSynthesis | undefined = globalThis.speechSynthesis,
): Speaker {
  let currentId = -1
  let spokenUpTo = 0
  const say = (sentence: string) => {
    if (!synth) return
    const utterance = new SpeechSynthesisUtterance(sentence)
    utterance.rate = 1.05
    synth.speak(utterance)
  }
  return {
    update(commentId, text, final) {
      if (!synth) return
      if (commentId !== currentId) {
        synth.cancel()
        currentId = commentId
        spokenUpTo = 0
      }
      const { sentences, next } = completedSentences(text, spokenUpTo)
      sentences.forEach(say)
      spokenUpTo = next
      if (final) {
        const rest = text.slice(spokenUpTo).trim()
        if (rest) say(rest)
        spokenUpTo = text.length
      }
    },
    cancel() {
      synth?.cancel()
      currentId = -1
      spokenUpTo = 0
    },
  }
}
