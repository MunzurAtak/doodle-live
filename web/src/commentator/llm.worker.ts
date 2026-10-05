/// <reference lib="webworker" />
/** Runs the WebLLM engine off the main thread (SPEC §8.1). */
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (event: MessageEvent) => handler.onmessage(event)
