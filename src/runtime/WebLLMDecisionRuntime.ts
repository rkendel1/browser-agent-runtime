import {
  DECISION_PROBABILITY_STATUS,
  DECISION_PROMPT_VERSION,
  DECISION_READOUT_TEMPERATURE,
  DECISION_SYSTEM_PROMPT,
  LOW_OPTION_MASS_THRESHOLD,
  assertValidDecisionInput,
  buildDecisionPrompt,
  estimateTokens,
  promptDigest,
  resolveOptionProbabilities,
  type DecisionInput,
  type DecisionResult,
  type DecisionRuntime,
  type LabelLogprob,
} from "./DecisionRuntime";
import {
  BENCHMARK_DECISION_MODEL,
  decisionAppConfig,
  findDecisionModel,
  type WebLLMAppConfig,
} from "./decisionModels";

/**
 * The slice of web-llm this runtime uses.
 *
 * Only `chat.completions.create` is needed, and only for its logprobs: the
 * request asks for a single token so the engine performs one forward pass and
 * reports the distribution at the answer position. The token it would have
 * sampled is never read.
 */
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

export interface WebLLMDecisionRuntimeOptions {
  /** A model id from `decisionModels.ts`, or any web-llm prebuilt model id. */
  model?: string;
  /**
   * How many of the top next-token candidates to request. web-llm caps this at
   * 5, which also caps how many options a single readout can separate.
   */
  topLogprobs?: number;
  appConfig?: WebLLMAppConfig;
  initProgressCallback?: (progress: { progress?: number; text?: string }) => void;
  createEngine?: () => Promise<WebLLMDecisionEngine>;
}

/** web-llm's ceiling for `top_logprobs`. */
export const MAX_TOP_LOGPROBS = 5;

/**
 * Reads a typed decision out of a WebGPU model's option-token logits.
 *
 * No JSON is generated, no explanation is generated, and no prose is parsed.
 * The engine runs one forward pass over the prompt; the decision is the softmax
 * over the option labels at the answer position.
 */
export class WebLLMDecisionRuntime implements DecisionRuntime {
  private enginePromise?: Promise<WebLLMDecisionEngine>;
  private execution = "webgpu";
  private readonly options: WebLLMDecisionRuntimeOptions;
  private readonly modelId: string;

  constructor(options: WebLLMDecisionRuntimeOptions = {}) {
    this.options = options;
    this.modelId = options.model ?? BENCHMARK_DECISION_MODEL.modelId;
  }

  async decide(input: DecisionInput): Promise<DecisionResult> {
    assertValidDecisionInput(input);

    const topLogprobs = Math.min(this.options.topLogprobs ?? MAX_TOP_LOGPROBS, MAX_TOP_LOGPROBS);
    if (input.options.length > topLogprobs) {
      // Say this rather than silently returning a decision over the subset of
      // options that happened to reach the top-k.
      console.warn(
        `[WebLLMDecisionRuntime] ${input.options.length} options but only the top ${topLogprobs} ` +
          "next-token candidates are observable; options outside that set read as probability 0.",
      );
    }

    const prompt = buildDecisionPrompt(input);
    const engine = await this.getEngine();
    const startedAt = now();
    const response = await engine.chat.completions.create({
      messages: [
        { role: "system", content: DECISION_SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      // One decode step, only so the answer position's logits are reported. The
      // sampled token is discarded, which is why the trace counts 0 generated
      // tokens: nothing the model wrote is read back.
      max_tokens: 1,
      // See DECISION_READOUT_TEMPERATURE: at 0 the reported distribution is
      // one-hot and the readout stops meaning anything.
      temperature: DECISION_READOUT_TEMPERATURE,
      logprobs: true,
      top_logprobs: topLogprobs,
      // Neutral penalties, so the reported distribution is the model's own.
      // web-llm only penalizes tokens it has already generated, and this
      // request generates none, but the readout should not depend on that.
      repetition_penalty: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
    });
    const latencyMs = now() - startedAt;

    const { probabilities, selected, optionMass } = resolveOptionProbabilities(
      input.options,
      readAnswerSlot(response),
    );

    return {
      selected,
      probabilities,
      latencyMs,
      trace: {
        model: this.modelId,
        execution: this.execution,
        readout: "option-logits",
        generatedTokens: 0,
        contextTokens:
          response.usage?.prompt_tokens ?? estimateTokens(DECISION_SYSTEM_PROMPT + prompt),
        options: input.options.length,
        selected,
        optionMass,
        lowOptionMass: optionMass < LOW_OPTION_MASS_THRESHOLD,
        probabilityStatus: DECISION_PROBABILITY_STATUS,
        promptVersion: DECISION_PROMPT_VERSION,
        promptSha256: await promptDigest(prompt),
        prompt,
      },
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
      // Only override the prebuilt catalogue for models this repo pins; an
      // arbitrary web-llm model id still resolves the usual way.
      appConfig: this.options.appConfig ?? (pinned ? decisionAppConfig([pinned]) : undefined),
    });
    await engine.reload(this.modelId);

    const vendor = await engine.getGPUVendor?.().catch(() => undefined);
    this.execution = vendor ? `webgpu (${vendor})` : "webgpu";
    return engine;
  }
}

/**
 * The candidate tokens at the answer position.
 *
 * `top_logprobs` holds the ranked alternatives; the chosen token is included
 * too, in case an engine reports it without repeating it in the list.
 */
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
    const alreadyListed = candidates.some((candidate) => candidate.token === slot.token);
    if (!alreadyListed) {
      candidates.push({ token: slot.token, logprob: slot.logprob });
    }
  }

  return candidates;
}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}
