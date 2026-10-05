/** Pick `n` distinct words for one game. `random` is injectable for tests. */
export function pickWords(
  categories: readonly string[],
  n: number,
  random: () => number = Math.random,
): string[] {
  if (n > categories.length) throw new Error(`need ${n} words, only ${categories.length} known`)
  const pool = [...categories]
  // Partial Fisher-Yates: the first n entries become a uniform random sample.
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(random() * (pool.length - i))
    ;[pool[i], pool[j]] = [pool[j], pool[i]]
  }
  return pool.slice(0, n)
}
