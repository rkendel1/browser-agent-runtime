import {
  DECISION_PROBABILITY_STATUS,
  DECISION_PROMPT_VERSION,
  DECISION_SYSTEM_PROMPT,
  LOW_OPTION_MASS_THRESHOLD,
  assertValidDecisionInput,
  buildDecisionPrompt,
  encodeCandidateLabels,
  estimateTokens,
  promptDigest,
  resolveOptionProbabilities,
  type BinaryDecisionInput,
  type BinaryDecisionResult,
  type ChoiceDecisionInput,
  type ChoiceDecisionResult,
  type DecisionInput,
  type DecisionProvenance,
  type DecisionResult,
  type DecisionRuntime,
  type DecisionRuntimeCapabilities,
  type ScoreDecisionInput,
  type ScoreDecisionResult,
} from "./DecisionRuntime";
import type { LocalLLMAdapter } from "./LocalLLMAdapter";

export interface LocalLLMDecisionRuntimeOptions {
  model?: string;
  modelRevision?: string;
  runtime?: string;
  runtimeVersion?: string;
  decisionSchemaRevision?: string;
}

export class LocalLLMDecisionRuntime implements DecisionRuntime {
  readonly capabilities: DecisionRuntimeCapabilities;
  private readonly options: LocalLLMDecisionRuntimeOptions;
  private readonly adapter: LocalLLMAdapter;

  constructor(
    adapter: LocalLLMAdapter,
    options: LocalLLMDecisionRuntimeOptions = {},
  ) {
    this.adapter = adapter;
    this.options = options;
    this.capabilities = adapter.capabilities;
  }

  async decide(input: DecisionInput): Promise<DecisionResult> {
    assertValidDecisionInput(input);
    const prompt = buildDecisionPrompt(input);
    const labels = input.options.map((option) => option.id);
    encodeCandidateLabels(labels);
    const startedAt = now();
    const response = await this.adapter.execute({
      prompt,
      systemPrompt: DECISION_SYSTEM_PROMPT,
      candidateLabels: input.options.map((_option, index) => String.fromCharCode(65 + index)),
      context: input.state,
    });
    const latencyMs = now() - startedAt;
    const { probabilities, selected, optionMass } = resolveOptionProbabilities(
      input.options,
      response.candidates,
    );
    const provenance: DecisionProvenance = {
      runtime: this.options.runtime ?? "local-llm",
      runtimeVersion: this.options.runtimeVersion ?? "unknown",
      model: response.model ?? this.options.model ?? "unknown",
      modelRevision: response.modelRevision ?? this.options.modelRevision ?? "unavailable",
      executionMethod: response.executionMethod,
      promptRevision: DECISION_PROMPT_VERSION,
      decisionSchemaRevision: this.options.decisionSchemaRevision ?? "decision-v1",
      browser: response.browser,
      webgpu: response.webgpu,
      gpu: response.gpu,
      quantization: response.quantization,
      webllmVersion: response.webllmVersion,
    };
    return {
      selected,
      probabilities,
      latencyMs,
      trace: {
        model: provenance.model,
        execution: response.gpu ? `webgpu (${response.gpu})` : response.executionMethod,
        readout: response.executionMethod === "generated" ? "generated-text" : "option-logits",
        generatedTokens: 0,
        contextTokens: response.promptTokens ?? estimateTokens(DECISION_SYSTEM_PROMPT + prompt),
        options: input.options.length,
        selected,
        optionMass,
        lowOptionMass: optionMass < LOW_OPTION_MASS_THRESHOLD,
        probabilityStatus: DECISION_PROBABILITY_STATUS,
        promptVersion: DECISION_PROMPT_VERSION,
        promptSha256: await promptDigest(prompt),
        prompt,
        provenance,
      },
    };
  }

  async decideBinary(input: BinaryDecisionInput): Promise<BinaryDecisionResult> {
    const result = await this.decide({
      ...input,
      options: [
        { id: "true", description: "true" },
        { id: "false", description: "false" },
      ],
    });
    return {
      ...result,
      selected: result.selected === "true",
      probabilities: {
        true: result.probabilities.true ?? 0,
        false: result.probabilities.false ?? 0,
      },
    };
  }

  decideChoice(input: ChoiceDecisionInput): Promise<ChoiceDecisionResult> {
    return this.decide({
      ...input,
      options: input.options.map((option) =>
        typeof option === "string" ? { id: option, description: option } : option,
      ),
    });
  }

  async decideScore(input: ScoreDecisionInput): Promise<ScoreDecisionResult> {
    const options = input.options ?? [0, 1, 2, 3, 4, 5];
    const result = await this.decide({
      ...input,
      options: options.map((score) => ({ id: String(score), description: String(score) })),
    });
    return { ...result, selected: Number(result.selected) };
  }
}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}
