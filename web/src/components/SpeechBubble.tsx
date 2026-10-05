interface SpeechBubbleProps {
  readonly text: string | null
  readonly streaming: boolean
  readonly placeholder: string
}

export function SpeechBubble({ text, streaming, placeholder }: SpeechBubbleProps) {
  const thinking = streaming && !text
  return (
    <div className="relative min-h-16 flex-1 rounded-2xl rounded-tl-sm border-2 border-ink bg-paper px-4 py-3">
      {thinking ? (
        <p className="flex h-7 items-center gap-1.5" aria-label="Thinking">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-2 w-2 animate-pulse rounded-full bg-ink-faint motion-reduce:animate-none"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </p>
      ) : (
        <p
          aria-live="polite"
          aria-busy={streaming}
          className={`text-lg leading-snug ${text ? 'text-ink' : 'text-ink-faint'}`}
        >
          {text ?? placeholder}
        </p>
      )}
    </div>
  )
}
