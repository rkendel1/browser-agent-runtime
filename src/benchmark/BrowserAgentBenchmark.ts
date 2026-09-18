import type {
  BrowserAgentRuntime,
  BrowserBlockReason,
  BrowserRunResult,
  BrowserTask,
} from "../browser/BrowserTypes";

export interface BrowserBenchmarkFixture {
  id: string;
  task: BrowserTask;
  /** Full expected decision sequence, making multi-step accuracy unambiguous. */
  expectedOptionIds: string[];
  expectedStatus?: BrowserRunResult["status"];
  expectedReason?: BrowserBlockReason;
  probe?: BrowserBenchmarkProbeName;
}

export type BrowserBenchmarkProbeName =
  | "option-order-invariance"
  | "option-label-invariance"
  | "snapshot-invalidation"
  | "empty-state"
  | "contradictory-state"
  | "irrelevant-state";

export interface BrowserBenchmarkMetadata {
  modelId: string;
  modelRevision: string;
  runtimeVersion: string;
  webllmVersion: string;
  browser: string;
  gpu: string;
  promptVersion: string;
  contextMode: "state" | "page" | "retrieved";
}

export interface BrowserBenchmarkRow {
  fixtureId: string;
  correct: boolean;
  selectedOptionIds: string[];
  expectedOptionIds: string[];
  status: BrowserRunResult["status"];
  reason?: BrowserBlockReason;
  decisionLatencyMs: number;
  executionLatencyMs: number;
  totalLatencyMs: number;
  steps: number;
  blockedDecisions: number;
  staleDecisions: number;
  invalidDecisions: number;
  generatedTokens: number;
  contextTokens: number;
}

export interface BrowserBenchmarkResult {
  metadata: BrowserBenchmarkMetadata & {
    reproducible: boolean;
    reproducibility: "PINNED" | "UNPINNED — not reproducible";
  };
  accuracy: number;
  probes: Record<BrowserBenchmarkProbeName, { runs: number; passed: number; rate?: number }>;
  rows: BrowserBenchmarkRow[];
  totals: Omit<
    BrowserBenchmarkRow,
    "fixtureId" | "correct" | "selectedOptionIds" | "expectedOptionIds" | "status" | "reason"
  >;
}

export async function runBrowserAgentBenchmark(
  runtime: BrowserAgentRuntime,
  fixtures: readonly BrowserBenchmarkFixture[],
  metadata: BrowserBenchmarkMetadata,
): Promise<BrowserBenchmarkResult> {
  const rows: BrowserBenchmarkRow[] = [];
  for (const fixture of fixtures) {
    const result = await runtime.run(fixture.task);
    const selectedOptionIds = result.evidence.map((entry) => entry.decision.optionId);
    const decisionLatencyMs = sum(result.evidence.map((entry) => entry.decision.latencyMs ?? 0));
    const executionLatencyMs = sum(result.evidence.map((entry) => entry.execution.durationMs));
    rows.push({
      fixtureId: fixture.id,
      correct:
        arraysEqual(selectedOptionIds, fixture.expectedOptionIds) &&
        (fixture.expectedStatus === undefined || result.status === fixture.expectedStatus) &&
        (fixture.expectedReason === undefined || result.reason === fixture.expectedReason),
      selectedOptionIds,
      expectedOptionIds: fixture.expectedOptionIds,
      status: result.status,
      reason: result.reason,
      decisionLatencyMs,
      executionLatencyMs,
      totalLatencyMs: result.latencyMs,
      steps: result.steps,
      blockedDecisions: result.evidence.filter((entry) => entry.execution.status === "blocked").length,
      staleDecisions: result.reason === "stale_decision" ? 1 : 0,
      invalidDecisions: isInvalid(result.reason) ? 1 : 0,
      generatedTokens: sum(result.evidence.map((entry) => entry.decision.generatedTokens ?? 0)),
      contextTokens: sum(result.evidence.map((entry) => entry.decision.contextTokens ?? 0)),
    });
  }
  const reproducible = /^[0-9a-f]{40}$/.test(metadata.modelRevision);
  return {
    metadata: {
      ...metadata,
      reproducible,
      reproducibility: reproducible ? "PINNED" : "UNPINNED — not reproducible",
    },
    accuracy: fixtures.length === 0 ? 0 : rows.filter((row) => row.correct).length / fixtures.length,
    probes: summarizeProbes(fixtures, rows),
    rows,
    totals: {
      decisionLatencyMs: sum(rows.map((row) => row.decisionLatencyMs)),
      executionLatencyMs: sum(rows.map((row) => row.executionLatencyMs)),
      totalLatencyMs: sum(rows.map((row) => row.totalLatencyMs)),
      steps: sum(rows.map((row) => row.steps)),
      blockedDecisions: sum(rows.map((row) => row.blockedDecisions)),
      staleDecisions: sum(rows.map((row) => row.staleDecisions)),
      invalidDecisions: sum(rows.map((row) => row.invalidDecisions)),
      generatedTokens: sum(rows.map((row) => row.generatedTokens)),
      contextTokens: sum(rows.map((row) => row.contextTokens)),
    },
  };
}

function summarizeProbes(
  fixtures: readonly BrowserBenchmarkFixture[],
  rows: readonly BrowserBenchmarkRow[],
): BrowserBenchmarkResult["probes"] {
  const names: BrowserBenchmarkProbeName[] = [
    "option-order-invariance",
    "option-label-invariance",
    "snapshot-invalidation",
    "empty-state",
    "contradictory-state",
    "irrelevant-state",
  ];
  return Object.fromEntries(
    names.map((name) => {
      const fixtureIds = new Set(fixtures.filter((fixture) => fixture.probe === name).map((fixture) => fixture.id));
      const probeRows = rows.filter((row) => fixtureIds.has(row.fixtureId));
      const passed = probeRows.filter((row) => row.correct).length;
      return [name, { runs: probeRows.length, passed, rate: probeRows.length ? passed / probeRows.length : undefined }];
    }),
  ) as BrowserBenchmarkResult["probes"];
}

/** Used by conformance and option-order/label invariance probes. */
export function decisionsConform(left: BrowserRunResult, right: BrowserRunResult): boolean {
  return arraysEqual(
    left.evidence.map((entry) => entry.decision.optionId),
    right.evidence.map((entry) => entry.decision.optionId),
  );
}

function isInvalid(reason: BrowserBlockReason | undefined): boolean {
  return reason !== undefined && [
    "unknown_option",
    "missing_element",
    "element_not_visible",
    "element_disabled",
    "invalid_role_action",
    "invalid_navigation",
    "missing_capability",
    "invalid_decision",
  ].includes(reason);
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function arraysEqual(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
