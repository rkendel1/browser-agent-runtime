import type { DecisionRuntime } from "../runtime/DecisionRuntime";
import { BrowserDecisionAdapter } from "./BrowserDecision";
import { observationByteLength } from "./BrowserContext";
import { validateAction, validateDecision, type BrowserExecutionPolicy } from "./BrowserValidation";
import type {
  BrowserAction,
  BrowserActionEvidence,
  BrowserAgentRuntime,
  BrowserBlockReason,
  BrowserDecision,
  BrowserDecisionRequest,
  BrowserExecutionResult,
  BrowserExecutor,
  BrowserObservation,
  BrowserRunResult,
  BrowserTask,
} from "./BrowserTypes";

export interface DefaultBrowserAgentRuntimeOptions extends BrowserExecutionPolicy {
  decisionPath?: "direct" | "generated";
  maxSteps?: number;
  maxLatency?: number;
  maxRetries?: number;
  maxObservationBytes?: number;
}

const DEFAULTS = {
  maxSteps: 20,
  maxLatency: 30_000,
  maxRetries: 0,
  maxObservationBytes: 256 * 1024,
};

export class DefaultBrowserAgentRuntime implements BrowserAgentRuntime {
  private readonly executor: BrowserExecutor;
  private readonly decisions: BrowserDecisionAdapter;
  private readonly options: DefaultBrowserAgentRuntimeOptions;

  constructor(
    executor: BrowserExecutor,
    decisionRuntime: DecisionRuntime,
    options: DefaultBrowserAgentRuntimeOptions = {},
  ) {
    this.executor = executor;
    this.decisions = new BrowserDecisionAdapter(decisionRuntime, { path: options.decisionPath });
    this.options = options;
  }

  observe(): Promise<BrowserObservation> {
    return this.executor.observe();
  }

  decide(request: BrowserDecisionRequest): Promise<BrowserDecision> {
    return this.decisions.decide(request);
  }

  async execute(
    action: BrowserAction,
    observation: BrowserObservation,
  ): Promise<BrowserExecutionResult> {
    const startedAt = now();
    let current: BrowserObservation;
    try {
      current = await this.executor.observe();
    } catch (error) {
      return failed(startedAt, error);
    }
    const validation = validateAction(action, observation, current, this.executor, this.options);
    if (!validation.ok) {
      return { status: "blocked", reason: validation.reason, durationMs: now() - startedAt };
    }
    if (action.type === "stop") {
      return { status: "complete", durationMs: now() - startedAt, resultingObservation: current };
    }
    try {
      await executeAction(this.executor, action);
      const actionFinishedAt = now();
      const resultingObservation = await this.executor.observe();
      return {
        status: "success",
        durationMs: actionFinishedAt - startedAt,
        resultingObservation,
      };
    } catch (error) {
      return failed(startedAt, error);
    }
  }

  async run(task: BrowserTask): Promise<BrowserRunResult> {
    const startedAt = now();
    const limits = resolveLimits(task, this.options);
    const evidence: BrowserActionEvidence[] = [];
    let observation: BrowserObservation;
    try {
      observation = await retry(() => this.observe(), limits.maxRetries);
    } catch (error) {
      return terminal("failed", "retry_limit", 0, startedAt, evidence, undefined, error);
    }

    for (let step = 0; step < limits.maxSteps; step += 1) {
      const sizeReason = checkObservation(observation, limits.maxObservationBytes);
      if (sizeReason) return terminal("blocked", sizeReason, step, startedAt, evidence, observation);
      if (now() - startedAt >= limits.maxLatency) {
        return terminal("blocked", "latency_limit", step, startedAt, evidence, observation);
      }

      const options =
        typeof task.options === "function" ? task.options(observation, step) : task.options;
      const request: BrowserDecisionRequest = {
        instruction: task.instruction,
        observation,
        options,
        context: {
          ...task.context,
          contextMode: task.contextMode ?? "state",
          retrievedContext: task.retrievedContext,
          maxContextBytes: limits.maxObservationBytes,
        },
      };
      let decision: BrowserDecision;
      try {
        decision = await retry(() => this.decide(request), limits.maxRetries);
      } catch (error) {
        return terminal("failed", "retry_limit", step, startedAt, evidence, observation, error);
      }
      if (now() - startedAt >= limits.maxLatency) {
        return terminal("blocked", "latency_limit", step, startedAt, evidence, observation);
      }

      let current: BrowserObservation;
      try {
        current = await retry(() => this.observe(), limits.maxRetries);
      } catch (error) {
        return terminal("failed", "retry_limit", step, startedAt, evidence, observation, error);
      }
      const currentSizeReason = checkObservation(current, limits.maxObservationBytes);
      if (currentSizeReason) {
        return terminal("blocked", currentSizeReason, step, startedAt, evidence, current);
      }
      const validation = validateDecision(decision, request, current, this.executor, this.options);
      const chosenAction = request.options.find((option) => option.id === decision.optionId)?.action;
      if (!validation.ok) {
        if (chosenAction) evidence.push(toEvidence(decision, chosenAction, "blocked", 0));
        return terminal("blocked", validation.reason, step + 1, startedAt, evidence, current);
      }

      const result = await this.executeCurrent(validation.action, current);
      evidence.push(
        toEvidence(
          decision,
          validation.action,
          result.status === "success" || result.status === "complete" ? "success" : result.status,
          result.durationMs,
          result.resultingObservation,
        ),
      );
      const resultingSizeReason = result.resultingObservation
        ? checkObservation(result.resultingObservation, limits.maxObservationBytes)
        : undefined;
      if (resultingSizeReason) {
        return terminal(
          "blocked",
          resultingSizeReason,
          step + 1,
          startedAt,
          evidence,
          result.resultingObservation,
        );
      }
      if (now() - startedAt >= limits.maxLatency) {
        return terminal(
          "blocked",
          "latency_limit",
          step + 1,
          startedAt,
          evidence,
          result.resultingObservation ?? current,
        );
      }
      if (result.status === "complete") {
        return terminal("complete", undefined, step + 1, startedAt, evidence, result.resultingObservation);
      }
      if (result.status === "blocked") {
        return terminal("blocked", result.reason, step + 1, startedAt, evidence, current);
      }
      if (result.status === "failed") {
        return terminal("failed", "execution_failed", step + 1, startedAt, evidence, current, result.error);
      }
      observation = result.resultingObservation ?? (await this.observe());
    }
    return terminal("blocked", "step_limit", limits.maxSteps, startedAt, evidence, observation);
  }

  private async executeCurrent(
    action: BrowserAction,
    current: BrowserObservation,
  ): Promise<BrowserExecutionResult> {
    const startedAt = now();
    if (action.type === "stop") {
      return { status: "complete", durationMs: 0, resultingObservation: current };
    }
    try {
      await executeAction(this.executor, action);
      const durationMs = now() - startedAt;
      const resultingObservation = await this.observe();
      return { status: "success", durationMs, resultingObservation };
    } catch (error) {
      return failed(startedAt, error);
    }
  }
}

async function executeAction(executor: BrowserExecutor, action: Exclude<BrowserAction, { type: "stop" }>) {
  switch (action.type) {
    case "click": return executor.click!(action.elementId);
    case "type": return executor.type!(action.elementId, action.value);
    case "select": return executor.select!(action.elementId, action.value);
    case "check": return executor.check!(action.elementId);
    case "uncheck": return executor.uncheck!(action.elementId);
    case "scroll": return executor.scroll!(action.direction, action.amount);
    case "navigate": return executor.navigate!(action.url);
    case "wait": return executor.wait!(action.ms);
  }
}

function resolveLimits(task: BrowserTask, options: DefaultBrowserAgentRuntimeOptions) {
  return {
    maxSteps: positiveInteger(task.maxSteps ?? options.maxSteps ?? DEFAULTS.maxSteps, "maxSteps"),
    maxLatency: positiveNumber(task.maxLatency ?? options.maxLatency ?? DEFAULTS.maxLatency, "maxLatency"),
    maxRetries: nonnegativeInteger(task.maxRetries ?? options.maxRetries ?? DEFAULTS.maxRetries, "maxRetries"),
    maxObservationBytes: positiveInteger(
      task.maxObservationBytes ?? options.maxObservationBytes ?? DEFAULTS.maxObservationBytes,
      "maxObservationBytes",
    ),
  };
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`);
  return value;
}

function nonnegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer.`);
  return value;
}

function positiveNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive.`);
  return value;
}

function checkObservation(observation: BrowserObservation, maxBytes: number): BrowserBlockReason | undefined {
  return observationByteLength(observation) > maxBytes ? "observation_limit" : undefined;
}

async function retry<T>(operation: () => Promise<T>, maxRetries: number): Promise<T> {
  let attempts = 0;
  while (true) {
    try {
      return await operation();
    } catch (error) {
      if (attempts >= maxRetries) throw error;
      attempts += 1;
    }
  }
}

function toEvidence(
  decision: BrowserDecision,
  action: BrowserAction,
  status: "success" | "blocked" | "failed",
  durationMs: number,
  resultingObservation?: BrowserObservation,
): BrowserActionEvidence {
  return {
    snapshotId: decision.evidence.snapshotId,
    decision: {
      optionId: decision.optionId,
      path: decision.path,
      latencyMs: decision.latencyMs,
      contextTokens: decision.evidence.contextTokens,
      generatedTokens: decision.evidence.generatedTokens,
      model: decision.model,
    },
    action,
    execution: { status, durationMs },
    resultingSnapshotId: resultingObservation?.provenance.snapshotId,
  };
}

function failed(startedAt: number, error: unknown): BrowserExecutionResult {
  return {
    status: "failed",
    reason: "execution_failed",
    durationMs: now() - startedAt,
    error: error instanceof Error ? error.message : String(error),
  };
}

function terminal(
  status: BrowserRunResult["status"],
  reason: BrowserBlockReason | undefined,
  steps: number,
  startedAt: number,
  evidence: BrowserActionEvidence[],
  finalObservation?: BrowserObservation,
  error?: unknown,
): BrowserRunResult {
  return {
    status,
    reason,
    steps,
    latencyMs: now() - startedAt,
    evidence,
    finalObservation,
    error: error === undefined ? undefined : error instanceof Error ? error.message : String(error),
  };
}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}
