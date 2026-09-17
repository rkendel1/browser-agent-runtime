import type { ModelRuntime } from "./ModelRuntime";
import {
  DECISION_PROBABILITY_STATUS,
  DECISION_PROMPT_VERSION,
  DECISION_SYSTEM_PROMPT,
  MAX_DECISION_OPTIONS,
  assertValidDecisionInput,
  buildDecisionPrompt,
  estimateTokens,
  optionLabel,
  promptDigest,
  DecisionReadoutError,
  type DecisionInput,
  type DecisionResult,
  type DecisionRuntime,
} from "./DecisionRuntime";

export interface GeneratedDecisionRuntimeOptions {
  model?: string;
  execution?: string;
}

/**
 * The comparison arm: the same decision, taken the ordinary way.
 *
 * The model writes an answer and the application parses a choice out of it.
 * This is kept as compact as an answer can be — one letter, no JSON, no
 * explanation — so the benchmark compares the two readouts rather than
 * comparing a logit read against a verbose chatbot.
 *
 * Note what the generated path cannot return: a distribution. A parsed answer
 * is one option at probability 1, which is the difference the benchmark exists
 * to show.
 */
export class GeneratedDecisionRuntime implements DecisionRuntime {
  private readonly model: ModelRuntime;
  private readonly options: GeneratedDecisionRuntimeOptions;

  constructor(model: ModelRuntime, options: GeneratedDecisionRuntimeOptions = {}) {
    this.model = model;
    this.options = options;
  }

  async decide(input: DecisionInput): Promise<DecisionResult> {
    assertValidDecisionInput(input);

    const prompt = buildDecisionPrompt(input);
    const startedAt = now();
    const answer = await this.model.generate({
      system: DECISION_SYSTEM_PROMPT,
      messages: [{ role: "user", content: prompt }],
    });
    const latencyMs = now() - startedAt;

    const selected = parseSelectedOption(answer, input);
    const probabilities: Record<string, number> = {};
    for (const option of input.options) {
      probabilities[option.id] = option.id === selected ? 1 : 0;
    }

    return {
      selected,
      probabilities,
      latencyMs,
      trace: {
        model: this.options.model ?? "unknown",
        execution: this.options.execution ?? "webgpu",
        readout: "generated-text",
        generatedTokens: estimateTokens(answer),
        contextTokens: estimateTokens(DECISION_SYSTEM_PROMPT + prompt),
        options: input.options.length,
        selected,
        // Left unset: a parsed answer carries no information about the mass the
        // model placed anywhere else.
        probabilityStatus: DECISION_PROBABILITY_STATUS,
        promptVersion: DECISION_PROMPT_VERSION,
        promptSha256: await promptDigest(prompt),
        prompt,
      },
    };
  }
}

/**
 * Pull a choice out of generated text.
 *
 * This is the parsing step the direct path does not have: a label anywhere in
 * the answer, or failing that an option id. It is lenient on purpose — a strict
 * parser would report the generated path as wrong when it was only chatty.
 */
export function parseSelectedOption(answer: string, input: DecisionInput): string {
  const labels = input.options.map((_, index) => optionLabel(index));
  const trimmed = answer.trim();

  const exact = labels.indexOf(trimmed.toUpperCase());
  if (exact >= 0) {
    return input.options[exact]!.id;
  }

  const labelMatch = trimmed
    .toUpperCase()
    .match(new RegExp(`\\b([A-${optionLabel(Math.min(input.options.length, MAX_DECISION_OPTIONS) - 1)}])\\b`));
  if (labelMatch) {
    const index = labels.indexOf(labelMatch[1]!);
    if (index >= 0) {
      return input.options[index]!.id;
    }
  }

  const lowercased = trimmed.toLowerCase();
  const byId = input.options.find((option) => lowercased.includes(option.id.toLowerCase()));
  if (byId) {
    return byId.id;
  }

  throw new DecisionReadoutError(
    `The generated answer named no declared option: ${JSON.stringify(trimmed.slice(0, 200))}`,
  );
}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}
