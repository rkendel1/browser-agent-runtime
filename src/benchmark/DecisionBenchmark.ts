import type { DecisionOption, DecisionResult, DecisionRuntime } from "../runtime/DecisionRuntime";

export interface DecisionFixture {
  id: string;
  name: string;
  category: string;
  state: string;
  question: string;
  options: DecisionOption[];
  /** The option id a careful reader would pick from the state. */
  expected: string;
}

export type DecisionPathName = "generated" | "direct";

export interface DecisionRunRow {
  fixtureId: string;
  path: DecisionPathName;
  selected?: string;
  expected: string;
  correct: boolean;
  latencyMs: number;
  generatedTokens: number;
  contextTokens: number;
  optionMass?: number;
  probabilities?: Record<string, number>;
  error?: string;
}

export interface DecisionPathSummary {
  path: DecisionPathName;
  runs: number;
  errors: number;
  correct: number;
  /** Errors count as incorrect: a decision that did not arrive is not accurate. */
  accuracy: number;
  medianLatencyMs: number;
  medianGeneratedTokens: number;
  medianContextTokens: number;
  totalGeneratedTokens: number;
}

export interface DecisionBenchmarkReport {
  model: string;
  fixtures: number;
  rows: DecisionRunRow[];
  summaries: DecisionPathSummary[];
  /**
   * Share of fixtures where both paths produced a decision and picked the same
   * option.
   *
   * This is a systems comparison, not a claim that the two readouts are
   * semantically equivalent — the same caution OpenJev's own benchmark carries.
   */
  agreement: number;
  comparedFixtures: number;
}

export interface DecisionBenchmarkPaths {
  generated?: DecisionRuntime;
  direct?: DecisionRuntime;
}

export interface DecisionBenchmarkOptions {
  fixtures: DecisionFixture[];
  paths: DecisionBenchmarkPaths;
  model: string;
  onProgress?: (progress: { completed: number; total: number; row: DecisionRunRow }) => void;
}

export async function runDecisionBenchmark(
  options: DecisionBenchmarkOptions,
): Promise<DecisionBenchmarkReport> {
  const entries = Object.entries(options.paths).filter(
    (entry): entry is [DecisionPathName, DecisionRuntime] => Boolean(entry[1]),
  );

  if (entries.length === 0) {
    throw new Error("The benchmark needs at least one decision path.");
  }

  const rows: DecisionRunRow[] = [];
  const total = options.fixtures.length * entries.length;

  for (const fixture of options.fixtures) {
    for (const [path, runtime] of entries) {
      const row = await runOne(fixture, path, runtime);
      rows.push(row);
      options.onProgress?.({ completed: rows.length, total, row });
    }
  }

  const summaries = entries.map(([path]) => summarizePath(rows, path));
  const { agreement, comparedFixtures } = measureAgreement(rows);

  return {
    model: options.model,
    fixtures: options.fixtures.length,
    rows,
    summaries,
    agreement,
    comparedFixtures,
  };
}

async function runOne(
  fixture: DecisionFixture,
  path: DecisionPathName,
  runtime: DecisionRuntime,
): Promise<DecisionRunRow> {
  try {
    const result: DecisionResult = await runtime.decide({
      state: fixture.state,
      question: fixture.question,
      options: fixture.options,
    });

    return {
      fixtureId: fixture.id,
      path,
      selected: result.selected,
      expected: fixture.expected,
      correct: result.selected === fixture.expected,
      latencyMs: result.latencyMs,
      generatedTokens: result.trace.generatedTokens,
      contextTokens: result.trace.contextTokens,
      optionMass: result.trace.optionMass,
      probabilities: result.probabilities,
    };
  } catch (error) {
    return {
      fixtureId: fixture.id,
      path,
      expected: fixture.expected,
      correct: false,
      latencyMs: 0,
      generatedTokens: 0,
      contextTokens: 0,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function summarizePath(rows: DecisionRunRow[], path: DecisionPathName): DecisionPathSummary {
  const pathRows = rows.filter((row) => row.path === path);
  const completed = pathRows.filter((row) => !row.error);
  const correct = pathRows.filter((row) => row.correct).length;

  return {
    path,
    runs: pathRows.length,
    errors: pathRows.length - completed.length,
    correct,
    accuracy: pathRows.length > 0 ? correct / pathRows.length : 0,
    medianLatencyMs: median(completed.map((row) => row.latencyMs)),
    medianGeneratedTokens: median(completed.map((row) => row.generatedTokens)),
    medianContextTokens: median(completed.map((row) => row.contextTokens)),
    totalGeneratedTokens: completed.reduce((sum, row) => sum + row.generatedTokens, 0),
  };
}

function measureAgreement(rows: DecisionRunRow[]): {
  agreement: number;
  comparedFixtures: number;
} {
  const byFixture = new Map<string, Partial<Record<DecisionPathName, string>>>();

  for (const row of rows) {
    if (!row.selected) {
      continue;
    }

    const entry = byFixture.get(row.fixtureId) ?? {};
    entry[row.path] = row.selected;
    byFixture.set(row.fixtureId, entry);
  }

  let compared = 0;
  let agreed = 0;

  for (const entry of byFixture.values()) {
    if (!entry.generated || !entry.direct) {
      continue;
    }

    compared += 1;
    if (entry.generated === entry.direct) {
      agreed += 1;
    }
  }

  return {
    agreement: compared > 0 ? agreed / compared : 0,
    comparedFixtures: compared,
  };
}

export function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0 ? (sorted[middle - 1]! + sorted[middle]!) / 2 : sorted[middle]!;
}

const PATH_LABELS: Record<DecisionPathName, string> = {
  generated: "Generated",
  direct: "Direct logits",
};

/** Markdown table, in the shape the PR asks for. */
export function formatDecisionBenchmarkTable(report: DecisionBenchmarkReport): string {
  const header = [
    "| Path | Accuracy | Median latency | Output tokens |",
    "| --- | --- | --- | --- |",
  ];

  const rows = report.summaries.map((summary) => {
    const accuracy = `${(summary.accuracy * 100).toFixed(0)}% (${summary.correct}/${summary.runs})`;
    const latency = `${Math.round(summary.medianLatencyMs)} ms`;
    return `| ${PATH_LABELS[summary.path]} | ${accuracy} | ${latency} | ${summary.totalGeneratedTokens} |`;
  });

  const footer = [
    "",
    `Model: ${report.model}`,
    `Fixtures: ${report.fixtures}`,
    report.comparedFixtures > 0
      ? `Paths agreed on ${(report.agreement * 100).toFixed(0)}% of ${report.comparedFixtures} compared fixtures ` +
        "(a systems comparison, not a claim that the two readouts are semantically equivalent)."
      : "Only one path ran, so there is no agreement figure.",
  ];

  return [...header, ...rows, ...footer].join("\n");
}
