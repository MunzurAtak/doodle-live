/** Which WebLLM build to load. IDs verified against `prebuiltAppConfig.model_list` of
 * @mlc-ai/web-llm 0.2.85. */
export type ModelSize = 'default' | 'light'

export const MODEL_OPTIONS: Record<ModelSize, { label: string; download: string }> = {
  default: { label: 'Qwen2.5 1.5B', download: '~1 GB' },
  light: { label: 'Qwen2.5 0.5B', download: '~0.4 GB' },
}

/** q4f16 builds need the GPU's `shader-f16` feature; otherwise use the f32 variant. */
export function chooseModelId(size: ModelSize, shaderF16: boolean): string {
  const params = size === 'default' ? '1.5B' : '0.5B'
  return `Qwen2.5-${params}-Instruct-${shaderF16 ? 'q4f16_1' : 'q4f32_1'}-MLC`
}

export type GpuSupport =
  | { readonly ok: true; readonly shaderF16: boolean }
  | { readonly ok: false; readonly reason: string }

interface GpuLike {
  requestAdapter(): Promise<{ features: { has(name: string): boolean } } | null>
}

export async function detectWebGPU(
  gpu: GpuLike | undefined = (globalThis.navigator as { gpu?: GpuLike } | undefined)?.gpu,
): Promise<GpuSupport> {
  if (!gpu) return { ok: false, reason: 'This browser has no WebGPU.' }
  try {
    const adapter = await gpu.requestAdapter()
    if (!adapter) return { ok: false, reason: 'No WebGPU-capable graphics card was found.' }
    return { ok: true, shaderF16: adapter.features.has('shader-f16') }
  } catch {
    return { ok: false, reason: 'WebGPU could not be started.' }
  }
}
