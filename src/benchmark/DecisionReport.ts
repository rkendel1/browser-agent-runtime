import type {
  BenchmarkEnvironment,
  BenchmarkModelRecord,
  BenchmarkRuntimeRecord,
  DecisionBenchmarkArtifact,
  DecisionBenchmarkResult,
  DecisionConditionSummary,
  DecisionContextSource,
  DecisionRunRow,
  ProbeRunRow,
} from "./DecisionBenchmark";

export interface BuildArtifactOptions {
  environment: BenchmarkEnvironment;
  model: BenchmarkModelRecord;
  runtime: BenchmarkRuntimeRecord;
  benchmark: DecisionBenchmarkResult;
  results: DecisionRunRow[];
  probes: ProbeRunRow[];
}

export function buildBenchmarkArtifact(
  options: BuildArtifactOptions,
): DecisionBenchmarkArtifact {
  return {
    environment: options.environment,
    model: options.model,
    runtime: options.runtime,
    benchmark: options.benchmark,
    results: options.results,
    probes: options.probes,
  };
}

const CONTEXT_LABELS: Record<DecisionContextSource, string> = {
  state: "State only (control)",
  "full-page": "Full page",
  retrieved: "Retrieved context",
};

const RULE = "────────────────────────────────────────────────────────────";

/**
 * The human-readable report.
 *
 * Direct probabilities are printed as probabilities and option mass, never as
 * "confidence": the two readouts answer different questions, and the report
 * keeps them in separate columns rather than merging them into one score.
 */
export function formatDecisionBenchmarkReport(artifact: DecisionBenchmarkArtifact): string {
  const lines: string[] = [
    "Browser Decision Benchmark",
    RULE,
    `Model:      ${artifact.model.modelId}`,
    `Revision:   ${artifact.model.revision ?? "(unknown)"}${
      artifact.model.pinned ? "" : "   ** UNPINNED — not reproducible **"
    }`,
    `web-llm:    ${artifact.model.webllmVersion}`,
    `Prompt:     ${artifact.runtime.promptVersion}`,
    `Runtime:    ${artifact.environment.engine === "webgpu" ? "WebGPU" : "STUB ENGINE (not a model result)"}`,
    `GPU:        ${formatGpu(artifact.environment)}`,
    `Browser:    ${artifact.environment.userAgent}`,
    `Fixtures:   ${artifact.benchmark.fixtureCount}`,
    `Run:        ${artifact.benchmark.startedAt} (${formatDuration(
      artifact.benchmark.durationMs,
    )})`,
    "",
  ];

  for (const context of contextsIn(artifact.benchmark)) {
    lines.push(...formatContextBlock(artifact, context), "");
  }

  lines.push(...formatMatrix(artifact), "");
  lines.push(...formatDiagnostics(artifact), "");
  lines.push(...formatProbes(artifact));

  return lines.join("\n");
}

function contextsIn(benchmark: DecisionBenchmarkResult): DecisionContextSource[] {
  const order: DecisionContextSource[] = ["state", "full-page", "retrieved"];
  const present = new Set(benchmark.conditions.map((condition) => condition.context));
  return order.filter((context) => present.has(context));
}

function formatContextBlock(
  artifact: DecisionBenchmarkArtifact,
  context: DecisionContextSource,
): string[] {
  const direct = findSummary(artifact, "direct", context);
  const generated = findSummary(artifact, "generated", context);
  const agreement = artifact.benchmark.agreements.find((entry) => entry.context === context);

  const row = (label: string, left: string, right: string) =>
    `${label.padEnd(22)}${left.padStart(11)}${right.padStart(15)}`;

  const lines = [
    CONTEXT_LABELS[context],
    row("", "Direct", "Generated"),
    row(
      "Accuracy",
      formatAccuracy(direct),
      formatAccuracy(generated),
    ),
    row(
      "Median latency",
      direct ? `${Math.round(direct.latencyMs.median)}ms` : "—",
      generated ? `${Math.round(generated.latencyMs.median)}ms` : "—",
    ),
    row(
      "Median output",
      direct ? `${direct.generatedTokens.median}` : "—",
      generated ? `${generated.generatedTokens.median}` : "—",
    ),
    row(
      "Median context",
      direct ? `${Math.round(direct.contextTokens.median)}` : "—",
      generated ? `${Math.round(generated.contextTokens.median)}` : "—",
    ),
    row(
      "Errors",
      direct ? `${direct.errors}` : "—",
      generated ? `${generated.errors}` : "—",
    ),
  ];

  if (agreement) {
    lines.push(
      row(
        "Agreement",
        `${formatPercent(agreement.agreement)}`,
        `(${agreement.agreed}/${agreement.comparedFixtures})`,
      ),
    );
  }

  if (direct?.optionMass) {
    lines.push(
      "Option mass (direct)",
      `  median              ${formatMass(direct.optionMass.median)}`,
      `  p10                 ${formatMass(direct.optionMass.p10)}`,
      `  p90                 ${formatMass(direct.optionMass.p90)}`,
      `  low-mass rows       ${direct.lowOptionMassCount}`,
    );
  }

  if (direct?.retrievalHitRate !== undefined || generated?.retrievalHitRate !== undefined) {
    const hitRate = direct?.retrievalHitRate ?? generated?.retrievalHitRate ?? 0;
    lines.push(`Retrieval held the state in ${formatPercent(hitRate)} of runs`);
  }

  return lines;
}

/** The experiment: context source down the side, readout across the top. */
function formatMatrix(artifact: DecisionBenchmarkArtifact): string[] {
  const contexts = contextsIn(artifact.benchmark);
  const lines = ["Accuracy matrix", `${"".padEnd(22)}${"Generated".padStart(11)}${"Direct".padStart(11)}`];

  for (const context of contexts) {
    lines.push(
      `${CONTEXT_LABELS[context].padEnd(22)}${formatAccuracy(
        findSummary(artifact, "generated", context),
      ).padStart(11)}${formatAccuracy(findSummary(artifact, "direct", context)).padStart(11)}`,
    );
  }

  return lines;
}

function formatDiagnostics(artifact: DecisionBenchmarkArtifact): string[] {
  const lines = ["Diagnostics"];

  for (const summary of artifact.benchmark.summaries) {
    lines.push(
      `  ${summary.conditionId.padEnd(24)} errors ${summary.errors}` +
        `, unreadable slots ${summary.unreadableSlots}` +
        `, low option mass ${summary.lowOptionMassCount}`,
    );
  }

  const failures = artifact.results.filter((row) => row.error);
  if (failures.length > 0) {
    lines.push("  Failures:");
    for (const row of failures.slice(0, 10)) {
      lines.push(`    ${row.fixtureId} ${row.conditionId}: ${row.error}`);
    }
    if (failures.length > 10) {
      lines.push(`    … ${failures.length - 10} more`);
    }
  }

  return lines;
}

function formatProbes(artifact: DecisionBenchmarkArtifact): string[] {
  if (artifact.probes.length === 0) {
    return ["Option mass probes: not run"];
  }

  const idWidth = Math.max(8, ...artifact.probes.map((probe) => probe.probeId.length)) + 2;
  const selectedWidth =
    Math.max(10, ...artifact.probes.map((probe) => (probe.selected ?? "error").length)) + 2;

  const lines = [
    "Option mass probes (no correct answer; watching mass, not accuracy)",
    `${"probe".padEnd(idWidth)}${"selected".padEnd(selectedWidth)}${"top p".padStart(8)}${"mass".padStart(
      8,
    )}  low`,
  ];

  for (const probe of artifact.probes) {
    const topProbability = probe.probabilities
      ? Math.max(...Object.values(probe.probabilities))
      : undefined;
    const selected = probe.selected ?? (probe.error ? "error" : "—");
    const top = topProbability === undefined ? "—" : topProbability.toFixed(2);
    const mass = probe.optionMass === undefined ? "—" : formatMass(probe.optionMass);

    lines.push(
      `${probe.probeId.padEnd(idWidth)}${selected.padEnd(selectedWidth)}${top.padStart(
        8,
      )}${mass.padStart(8)}  ${probe.lowOptionMass ? "yes" : "no"}`,
    );
  }

  return lines;
}

function findSummary(
  artifact: DecisionBenchmarkArtifact,
  path: "direct" | "generated",
  context: DecisionContextSource,
): DecisionConditionSummary | undefined {
  return artifact.benchmark.summaries.find(
    (summary) => summary.path === path && summary.context === context,
  );
}

function formatAccuracy(summary: DecisionConditionSummary | undefined): string {
  return summary ? `${formatPercent(summary.accuracy)}` : "—";
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function formatMass(value: number): string {
  return value.toFixed(2).replace(/^0/, "");
}

function formatDuration(durationMs: number): string {
  const seconds = Math.round(durationMs / 1000);
  return seconds < 90 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function formatGpu(environment: BenchmarkEnvironment): string {
  if (!environment.gpu) {
    return environment.webgpu ? "WebGPU adapter, no reported info" : "no WebGPU adapter";
  }

  const { vendor, architecture, device, description } = environment.gpu;
  return [vendor, architecture, device, description].filter(Boolean).join(" · ") || "unknown";
}

/** Markdown table for pasting into a PR or README. */
export function formatDecisionBenchmarkTable(artifact: DecisionBenchmarkArtifact): string {
  const header = [
    "| Context | Path | Accuracy | Median latency | Output tokens | Option mass (median) |",
    "| --- | --- | --- | --- | --- | --- |",
  ];

  const rows = artifact.benchmark.summaries.map((summary) => {
    const accuracy = `${formatPercent(summary.accuracy)} (${summary.correct}/${summary.runs})`;
    const mass = summary.optionMass ? summary.optionMass.median.toFixed(2) : "—";
    return `| ${CONTEXT_LABELS[summary.context]} | ${summary.path} | ${accuracy} | ${Math.round(
      summary.latencyMs.median,
    )} ms | ${summary.totalGeneratedTokens} | ${mass} |`;
  });

  return [...header, ...rows].join("\n");
}
