interface SpeechBubbleProps {
  readonly text: string | null
  readonly placeholder: string
}

export function SpeechBubble({ text, placeholder }: SpeechBubbleProps) {
  return (
    <div className="relative min-h-16 flex-1 rounded-2xl rounded-tl-sm border-2 border-ink bg-paper px-4 py-3">
      <p
        aria-live="polite"
        className={`text-lg leading-snug ${text ? 'text-ink' : 'text-ink-faint'}`}
      >
        {text ?? placeholder}
      </p>
    </div>
  )
}
