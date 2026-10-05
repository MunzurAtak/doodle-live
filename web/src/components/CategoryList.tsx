export function CategoryList({ categories }: { readonly categories: readonly string[] }) {
  return (
    <details className="text-sm text-ink-soft">
      <summary className="cursor-pointer font-bold text-ink hover:text-marker">
        The {categories.length} things I can recognise
      </summary>
      <p className="mt-2 leading-relaxed">{categories.join(', ')}.</p>
    </details>
  )
}
