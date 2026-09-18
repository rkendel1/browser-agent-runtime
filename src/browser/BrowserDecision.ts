import type { DecisionResult, DecisionRuntime } from "../runtime/DecisionRuntime";
import { projectBrowserContext } from "./BrowserContext";
import type {
  BrowserAction,
  BrowserDecision,
  BrowserDecisionOption,
  BrowserDecisionRequest,
  BrowserDecisionRuntimeOptions,
  ConfidenceStatus,
} from "./BrowserTypes";

export class BrowserDecisionAdapter {
  private readonly runtime: DecisionRuntime;
  private readonly options: BrowserDecisionRuntimeOptions;

  constructor(runtime: DecisionRuntime, options: BrowserDecisionRuntimeOptions = {}) {
    this.runtime = runtime;
    this.options = options;
  }

  async decide(request: BrowserDecisionRequest): Promise<BrowserDecision> {
    assertBrowserDecisionRequest(request);
    const projection = projectBrowserContext(request.observation, {
      mode: readContextMode(request.context),
      retrievedContext: request.context?.retrievedContext,
      maxBytes: readMaxContextBytes(request.context),
    });
    const result = await this.runtime.decide({
      state: projection.serialized,
      question: request.instruction,
      options: request.options.map(({ id, description }) => ({ id, description })),
    });
    return normalizeBrowserDecision(result, request, this.options.path);
  }
}

export function normalizeBrowserDecision(
  result: DecisionResult,
  request: BrowserDecisionRequest,
  expectedPath?: "direct" | "generated",
): BrowserDecision {
  if (!request.options.some((option) => option.id === result.selected)) {
    throw new Error(`Decision runtime returned unknown option: ${result.selected}`);
  }
  const path = result.trace.readout === "generated-text" ? "generated" : "direct";
  if (expectedPath && expectedPath !== path) {
    throw new Error(`Expected ${expectedPath} decision readout, received ${path}.`);
  }
  const confidence = result.probabilities[result.selected];
  const confidenceStatus = diagnosticStatus(confidence, path === "generated");
  const optionMassStatus =
    result.trace.optionMassStatus ?? diagnosticStatus(result.trace.optionMass, path === "generated");
  const provenance = result.trace.provenance;
  return {
    optionId: result.selected,
    confidence: confidenceStatus === "available" ? confidence : undefined,
    confidenceStatus,
    optionMass: optionMassStatus === "available" ? result.trace.optionMass : undefined,
    optionMassStatus,
    path,
    model: {
      id: provenance?.model ?? result.trace.model,
      revision: provenance?.modelRevision,
      runtime: provenance?.runtime ?? result.trace.execution,
    },
    evidence: {
      snapshotId: request.observation.provenance.snapshotId,
      contextTokens: result.trace.contextTokens,
      generatedTokens: result.trace.generatedTokens,
    },
    latencyMs: result.latencyMs,
  };
}

export function normalizeDirectDecision(
  result: DecisionResult,
  request: BrowserDecisionRequest,
): BrowserDecision {
  return normalizeBrowserDecision(result, request, "direct");
}

export function normalizeGeneratedDecision(
  result: DecisionResult,
  request: BrowserDecisionRequest,
): BrowserDecision {
  return normalizeBrowserDecision(result, request, "generated");
}

export function deterministicBrowserOptionId(action: BrowserAction, description = ""): string {
  const source = stableStringify({ action, description: description.trim() });
  let hash = 0x811c9dc5;
  for (let index = 0; index < source.length; index += 1) {
    hash ^= source.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `option-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

export function browserDecisionOption(
  description: string,
  action: BrowserAction,
  id = deterministicBrowserOptionId(action, description),
): BrowserDecisionOption {
  return { id, description, action };
}

function diagnosticStatus(value: number | undefined, unsupported: boolean): ConfidenceStatus {
  if (unsupported) return "unsupported";
  if (value === undefined) return "unavailable";
  return Number.isFinite(value) && value >= 0 && value <= 1 ? "available" : "invalid";
}

function assertBrowserDecisionRequest(request: BrowserDecisionRequest): void {
  if (!request.instruction.trim()) throw new Error("A browser decision needs an instruction.");
  if (request.options.length < 2) throw new Error("A browser decision needs at least two options.");
  const ids = new Set<string>();
  for (const option of request.options) {
    if (!option.id.trim() || ids.has(option.id)) throw new Error(`Invalid option id: ${option.id}`);
    if (!option.description.trim()) throw new Error(`Option ${option.id} needs a description.`);
    ids.add(option.id);
  }
}

function readContextMode(context: Record<string, unknown> | undefined): "state" | "page" | "retrieved" {
  const mode = context?.contextMode;
  return mode === "page" || mode === "retrieved" ? mode : "state";
}

function readMaxContextBytes(context: Record<string, unknown> | undefined): number {
  const value = context?.maxContextBytes;
  return typeof value === "number" ? value : 64 * 1024;
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stableStringify(entry)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}
