/**
 * A decision runtime answers a closed question by reading the model's own
 * distribution over the declared options instead of generating an answer.
 *
 * This is deliberately a second runtime beside `ModelRuntime`: generation and
 * decision are different interfaces between the model and the application, and
 * collapsing them into one abstraction hides the difference this repository is
 * trying to measure.
 *
 * The technique follows OpenJev's direct mode
 * (https://github.com/rkendel1/open-jev, `src/openjev_phase1/direct.py`): a
 * single forward pass over the declared options, softmax over the answer-slot
 * logits, and no sampled answer token.
 */

export interface DecisionOption {
  id: string;
  description: string;
}

export interface DecisionInput {
  state: string;
  question: string;
  options: DecisionOption[];
}

/**
 * How a decision was read out of the model.
 *
 * - `option-logits`: the decision comes from the next-token distribution at the
 *   answer position, restricted to the option labels. No answer token is parsed.
 * - `generated-text`: the model wrote an answer and the application parsed it.
 */
export type DecisionReadout = "option-logits" | "generated-text";

export interface DecisionTrace {
  model: string;
  execution: string;
  readout: DecisionReadout;
  /**
   * Answer tokens generated and parsed by the application.
   *
   * The direct path reports 0: the readout inspects the logits at the answer
   * position, and whatever token the sampler would have produced there is
   * discarded rather than parsed.
   */
  generatedTokens: number;
  contextTokens: number;
  options: number;
  selected: string;
  /**
   * Probability mass on the option labels before renormalization — OpenJev's
   * `allowed_token_mass`.
   *
   * A low value means the model wanted to say something other than an option
   * label at the answer position, so the normalized probabilities are a
   * confident-looking readout of a model that was not answering the question.
   *
   * Undefined when the readout cannot measure it, as a parsed answer cannot.
   */
  optionMass?: number;
  /**
   * Diagnostic marker: `optionMass` fell below `LOW_OPTION_MASS_THRESHOLD`.
   *
   * Nothing in this repository behaves differently when it is true. It exists
   * so reports can count how often the phenomenon shows up, before anyone
   * decides what an agent should do about it.
   */
  lowOptionMass?: boolean;
  /**
   * Carried verbatim from OpenJev: these numbers rank the options against each
   * other under this prompt. They are not calibrated decision confidence.
   */
  probabilityStatus: string;
  promptVersion: string;
  promptSha256: string;
  prompt: string;
  provenance?: DecisionProvenance;
}

export interface DecisionProvenance {
  runtime: string;
  runtimeVersion: string;
  model: string;
  modelRevision: string;
  executionMethod: "logit" | "structured" | "generated" | "head";
  promptRevision: string;
  decisionSchemaRevision: string;
  browser?: string;
  webgpu?: boolean;
  gpu?: string;
  quantization?: string;
  webllmVersion?: string;
}

export interface DecisionResult {
  selected: string;
  probabilities: Record<string, number>;
  latencyMs: number;
  trace: DecisionTrace;
}

export interface DecisionRuntime {
  decide(input: DecisionInput): Promise<DecisionResult>;
}

export interface DecisionRuntimeCapabilities {
  binary: boolean;
  choice: boolean;
  score: boolean;
  batch: boolean;
  distribution: boolean;
  executionMethods: Array<DecisionProvenance["executionMethod"]>;
}

export type BinaryDecisionInput = Omit<DecisionInput, "options">;
export type ChoiceDecisionInput = Omit<DecisionInput, "options"> & {
  options: Array<DecisionOption | string>;
};
export type ScoreDecisionInput = Omit<DecisionInput, "options"> & {
  options?: number[];
};

export interface BinaryDecisionResult extends Omit<DecisionResult, "selected" | "probabilities"> {
  selected: boolean;
  probabilities: Record<"true" | "false", number>;
}

export interface ChoiceDecisionResult extends DecisionResult {
  selected: string;
}

export interface ScoreDecisionResult extends Omit<DecisionResult, "selected"> {
  selected: number;
}

export function encodeCandidateLabels(options: readonly string[]): Map<string, string> {
  if (options.length < 2 || options.length > MAX_DECISION_OPTIONS) {
    throw new Error(`A decision supports between 2 and ${MAX_DECISION_OPTIONS} options.`);
  }

  const labels = new Map<string, string>();
  for (const [index, option] of options.entries()) {
    if (!option.trim()) {
      throw new Error("Every candidate option needs a value.");
    }
    if (labels.has(option)) {
      throw new Error(`Duplicate candidate option: ${option}`);
    }
    labels.set(option, optionLabel(index));
  }
  return labels;
}

export class DecisionBatch {
  constructor(private readonly runtime: DecisionRuntime) {}

  decide(inputs: readonly DecisionInput[]): Promise<DecisionResult[]> {
    return Promise.all(inputs.map((input) => this.runtime.decide(input)));
  }
}

/** Raised when the readout cannot be tied back to any declared option. */
export class DecisionReadoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DecisionReadoutError";
  }
}

export const DECISION_PROMPT_VERSION = "browser-direct-options-v1";

/**
 * Provisional reporting threshold for `lowOptionMass`.
 *
 * Chosen to bucket rows in a report, not to gate a decision: half the
 * next-token mass going somewhere other than the option labels is enough to
 * want a second look. It is not calibrated, and no runtime reads it to change
 * what it returns.
 */
export const LOW_OPTION_MASS_THRESHOLD = 0.5;

/**
 * Temperature for the direct readout. It must be 1, and that is not a style
 * choice.
 *
 * web-llm reports `top_logprobs` from the distribution it built for sampling:
 * `softmax(logits / max(temperature, 1e-6))`. At `temperature: 0` that clamp
 * turns the readout into a one-hot vector — every decision comes back at 100%
 * against 0%, and option mass collapses to "was the argmax a label", which is
 * not a measurement of anything. At temperature 1 the reported distribution is
 * the model's own next-token distribution, which is what OpenJev reads off the
 * raw logits.
 *
 * The sampled token is still discarded, so a non-zero temperature introduces no
 * randomness into the decision: the softmax over the option labels is
 * deterministic.
 */
export const DECISION_READOUT_TEMPERATURE = 1;

export const DECISION_PROBABILITY_STATUS =
  "conditional option score; uncalibrated as decision confidence";

/**
 * Mirrors OpenJev's `DIRECT_SYSTEM`. The model is never asked for JSON, an
 * explanation, or prose — only for the answer slot to be an option label.
 */
export const DECISION_SYSTEM_PROMPT =
  "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. " +
  "Respond with only its uppercase letter, with no explanation or reasoning.";

const OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"] as const;

export const MAX_DECISION_OPTIONS = OPTION_LABELS.length;

export function optionLabel(index: number): string {
  const label = OPTION_LABELS[index];
  if (!label) {
    throw new Error(`Option index ${index} is out of range.`);
  }

  return label;
}

export function assertValidDecisionInput(input: DecisionInput): void {
  if (!input.question.trim()) {
    throw new Error("A decision needs a question.");
  }

  if (input.options.length < 2) {
    throw new Error("A decision needs at least two options.");
  }

  if (input.options.length > MAX_DECISION_OPTIONS) {
    throw new Error(`A decision supports at most ${MAX_DECISION_OPTIONS} options.`);
  }

  const seen = new Set<string>();
  for (const option of input.options) {
    if (!option.id.trim()) {
      throw new Error("Every option needs an id.");
    }

    if (!option.description.trim()) {
      throw new Error(`Option ${option.id} needs a description.`);
    }

    if (seen.has(option.id)) {
      throw new Error(`Duplicate option id: ${option.id}`);
    }

    seen.add(option.id);
  }
}

/**
 * The decision prompt is fixed and deterministic, so the only things that change
 * between runs are the state, the question and the options.
 *
 * It is sent as the user turn; the chat template's generation prefix puts the
 * answer slot at the very next token position.
 */
export function buildDecisionPrompt(input: DecisionInput): string {
  const options = input.options
    .map((option, index) => `${optionLabel(index)}. ${collapseWhitespace(option.description)}`)
    .join("\n");

  return [
    "State:",
    input.state.trim() || "(no state provided)",
    "Question:",
    collapseWhitespace(input.question),
    "Options:",
    options,
    "Select the option that best answers the question.",
  ].join("\n");
}

export interface LabelLogprob {
  token: string;
  logprob: number;
}

export interface OptionProbabilities {
  selected: string;
  probabilities: Record<string, number>;
  optionMass: number;
}

/**
 * Turn the next-token distribution into a distribution over the declared
 * options.
 *
 * Only tokens that are an option label count. Everything else the model might
 * have said at the answer position is dropped, and `optionMass` records how much
 * probability that dropped remainder held.
 */
export function resolveOptionProbabilities(
  options: DecisionOption[],
  candidates: LabelLogprob[],
): OptionProbabilities {
  const logprobByOptionId = new Map<string, number>();
  let optionMass = 0;

  for (const candidate of candidates) {
    const index = matchOptionIndex(candidate.token, options.length);
    if (index === undefined) {
      continue;
    }

    const optionId = options[index]!.id;

    // A label can reach the answer slot under more than one spelling ("A" and
    // " A"). Every spelling counts toward the allowed-token mass; the strongest
    // one decides the ranking.
    optionMass += Math.exp(candidate.logprob);
    const existing = logprobByOptionId.get(optionId);
    if (existing === undefined || candidate.logprob > existing) {
      logprobByOptionId.set(optionId, candidate.logprob);
    }
  }

  if (logprobByOptionId.size === 0) {
    throw new DecisionReadoutError(
      `None of the ${options.length} option labels appeared in the returned logits (saw: ${candidates
        .map((candidate) => JSON.stringify(candidate.token))
        .join(", ")}).`,
    );
  }

  const probabilities = softmaxOverOptions(options, logprobByOptionId);
  const selected = pickHighest(options, probabilities);

  return { selected, probabilities, optionMass: Math.min(optionMass, 1) };
}

function softmaxOverOptions(
  options: DecisionOption[],
  logprobByOptionId: Map<string, number>,
): Record<string, number> {
  const maximum = Math.max(...logprobByOptionId.values());
  const weights = new Map<string, number>();
  let total = 0;

  for (const [optionId, logprob] of logprobByOptionId) {
    const weight = Math.exp(logprob - maximum);
    weights.set(optionId, weight);
    total += weight;
  }

  const probabilities: Record<string, number> = {};
  for (const option of options) {
    const weight = weights.get(option.id) ?? 0;
    probabilities[option.id] = total > 0 ? weight / total : 0;
  }

  return probabilities;
}

function pickHighest(options: DecisionOption[], probabilities: Record<string, number>): string {
  let selected = options[0]!.id;

  for (const option of options) {
    if ((probabilities[option.id] ?? 0) > (probabilities[selected] ?? 0)) {
      selected = option.id;
    }
  }

  return selected;
}

/**
 * Match a raw token against an option label.
 *
 * OpenJev can check answer slots against the tokenizer directly; web-llm hands
 * back token strings, so the answer slot is matched by spelling. A label may
 * arrive as `"A"`, `" A"`, `"A."` or `"A)"` depending on the tokenizer, and all
 * of those read as the same option. Anything longer is a word that merely starts
 * with the letter, not the answer slot.
 */
function matchOptionIndex(token: string, optionCount: number): number | undefined {
  const trimmed = token.trim();
  if (!trimmed) {
    return undefined;
  }

  const head = trimmed[0]!;
  const rest = trimmed.slice(1);
  if (rest && !/^[.):\-]$/.test(rest)) {
    return undefined;
  }

  const index = OPTION_LABELS.indexOf(head as (typeof OPTION_LABELS)[number]);
  return index >= 0 && index < optionCount ? index : undefined;
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Digest of the exact prompt text, so a benchmark run can prove which prompt
 * produced it (OpenJev records `prompt_sha256` for the same reason).
 */
export async function promptDigest(prompt: string): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    return "";
  }

  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(prompt));
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function collapseWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}
