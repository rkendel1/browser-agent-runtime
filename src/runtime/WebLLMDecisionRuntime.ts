import {
  LocalLLMDecisionRuntime,
  type LocalLLMDecisionRuntimeOptions,
} from "./LocalLLMDecisionRuntime";
import {
  WebLLMAdapter,
  MAX_TOP_LOGPROBS,
  type WebLLMAdapterOptions,
  type WebLLMDecisionEngine,
} from "./WebLLMAdapter";

export { MAX_TOP_LOGPROBS, WebLLMAdapter };
export type { WebLLMAdapterOptions, WebLLMDecisionEngine };

export interface WebLLMDecisionRuntimeOptions extends WebLLMAdapterOptions, LocalLLMDecisionRuntimeOptions {}

export class WebLLMDecisionRuntime extends LocalLLMDecisionRuntime {
  constructor(options: WebLLMDecisionRuntimeOptions = {}) {
    const { model, topLogprobs, appConfig, initProgressCallback, createEngine, ...runtimeOptions } = options;
    super(
      new WebLLMAdapter({
        model,
        topLogprobs,
        appConfig,
        initProgressCallback,
        createEngine,
      }),
      runtimeOptions,
    );
  }
}
