import {
  LOCAL_LLM_DECISION_CAPABILITIES,
  type LocalLLMAdapter,
  type LocalLLMExecutionRequest,
  type LocalLLMExecutionResult,
} from "./LocalLLMAdapter";
import {
  DECISION_READOUT_TEMPERATURE,
  type LabelLogprob,
} from "./DecisionRuntime";
import {
  BENCHMARK_DECISION_MODEL,
  WEB_LLM_VERSION,
  decisionAppConfig,
  findDecisionModel,
  type WebLLMAppConfig,
} from "./decisionModels";

export interface WebLLMDecisionEngine {
  reload(model: string): Promise<void>;
  getGPUVendor?(): Promise<string>;
  chat: {
    completions: {
      create(request: {
        messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
        max_tokens?: number;
        temperature?: number;
        logprobs?: boolean;
        top_logprobs?: number;
        repetition_penalty?: number;
        frequency_penalty?: number;
        presence_penalty?: number;
      }): Promise<WebLLMChatCompletion>;
    };
  };
}

interface WebLLMChatCompletion {
  choices?: Array<{
    logprobs?: {
      content?: Array<{
        token?: string;
        logprob?: number;
        top_logprobs?: Array<{ token?: string; logprob?: number }>;
      }> | null;
    } | null;
  }>;
  usage?: { prompt_tokens?: number };
}

export interface WebLLMAdapterOptions {
  model?: string;
  topLogprobs?: number;
  appConfig?: WebLLMAppConfig;
  initProgressCallback?: (progress: { progress?: number; text?: string }) => void;
  createEngine?: () => Promise<WebLLMDecisionEngine>;
}

export const MAX_TOP_LOGPROBS = 5;

export class WebLLMAdapter implements LocalLLMAdapter {
  readonly capabilities = LOCAL_LLM_DECISION_CAPABILITIES;
  private enginePromise?: Promise<WebLLMDecisionEngine>;
  private gpu?: string;
  private readonly modelId: string;

  constructor(private readonly options: WebLLMAdapterOptions = {}) {
    this.modelId = options.model ?? BENCHMARK_DECISION_MODEL.modelId;
  }

  async execute(request: LocalLLMExecutionRequest): Promise<LocalLLMExecutionResult> {
    const topLogprobs = Math.min(this.options.topLogprobs ?? MAX_TOP_LOGPROBS, MAX_TOP_LOGPROBS);
    const engine = await this.getEngine();
    const response = await engine.chat.completions.create({
      messages: [
        { role: "system", content: request.systemPrompt },
        { role: "user", content: request.prompt },
      ],
      max_tokens: 1,
      temperature: DECISION_READOUT_TEMPERATURE,
      logprobs: true,
      top_logprobs: topLogprobs,
      repetition_penalty: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
    });

    return {
      candidates: readAnswerSlot(response),
      promptTokens: response.usage?.prompt_tokens,
      executionMethod: "logit",
      model: this.modelId,
      modelRevision: findDecisionModel(this.modelId)?.revision ?? "unavailable",
      browser: globalThis.navigator?.userAgent,
      webgpu: Boolean(globalThis.navigator?.gpu),
      gpu: this.gpu,
      webllmVersion: WEB_LLM_VERSION,
    };
  }

  private getEngine(): Promise<WebLLMDecisionEngine> {
    if (!this.enginePromise) {
      this.enginePromise = this.options.createEngine?.() ?? this.createDefaultEngine();
    }
    return this.enginePromise;
  }

  private async createDefaultEngine(): Promise<WebLLMDecisionEngine> {
    const webllm = (await import("@mlc-ai/web-llm")) as unknown as {
      MLCEngine: new (config?: unknown) => WebLLMDecisionEngine;
    };
    const pinned = findDecisionModel(this.modelId);
    const engine = new webllm.MLCEngine({
      initProgressCallback: this.options.initProgressCallback,
      appConfig: this.options.appConfig ?? (pinned ? decisionAppConfig([pinned]) : undefined),
    });
    await engine.reload(this.modelId);
    this.gpu = await engine.getGPUVendor?.().catch(() => undefined);
    return engine;
  }
}

function readAnswerSlot(response: WebLLMChatCompletion): LabelLogprob[] {
  const slot = response.choices?.[0]?.logprobs?.content?.[0];
  if (!slot) {
    throw new Error(
      "The engine returned no logprobs for the answer position. The direct decision path " +
        "requires a runtime that reports token logprobs.",
    );
  }
  const candidates: LabelLogprob[] = (slot.top_logprobs ?? [])
    .filter((entry): entry is { token: string; logprob: number } =>
      typeof entry.token === "string" && typeof entry.logprob === "number",
    )
    .map((entry) => ({ token: entry.token, logprob: entry.logprob }));
  if (typeof slot.token === "string" && typeof slot.logprob === "number") {
    if (!candidates.some((candidate) => candidate.token === slot.token)) {
      candidates.push({ token: slot.token, logprob: slot.logprob });
    }
  }
  return candidates;
}
