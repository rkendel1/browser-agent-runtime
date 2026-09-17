import type {
  DecisionOption,
  DecisionResult,
  DecisionRuntime,
} from "../runtime/DecisionRuntime";

export interface DecisionFixture {
  id: string;
  name: string;
  category: string;
  /** The decision's evidence, stated directly. */
  state: string;
  question: string;
  options: DecisionOption[];
  /** The option id a careful reader would pick from the state. */
  expected: string;
  /** Retrieval query for the page below. */
  query: string;
  /** A page that contains the state among distractor sections. */
  pageHtml: string;
}

/** A decision with no correct answer, used only to observe option mass. */
export interface OptionMassProbe {
  id: string;
  name: string;
  why: string;
  state: string;
  question: string;
  options: DecisionOption[];
}

export type DecisionPathName = "generated" | "direct";

/**
 * Where the state handed to the model comes from.
 *
 * - `state`: the fixture's own statement — the control, with no page and no
 *   retrieval in the way.
 * - `full-page`: every word of the fixture's page.
 * - `retrieved`: the chunks `search_context` returns for the fixture's query.
 */
export type DecisionContextSource = "state" | "full-page" | "retrieved";

export interface DecisionCondition {
  path: DecisionPathName;
  context: DecisionContextSource;
  runtime: DecisionRuntime;
}

export function conditionId(condition: {
  path: DecisionPathName;
  context: DecisionContextSource;
}): string {
  return `${condition.path}/${condition.context}`;
}

/** The state for one fixture under one context source, resolved by the caller. */
export interface ResolvedState {
  state: string;
  /** Chunk ids `search_context` returned, for retrieved context only. */
  retrievedChunkIds?: string[];
  /**
   * Whether the retrieved context actually contains the fixture's state.
   *
   * Separates a retrieval miss from a decision miss: a wrong answer over
   * context that never held the evidence is not the decision path's failure.
   */
  retrievedStateHit?: boolean;
}

export type StateResolver = (
  fixture: DecisionFixture,
  context: DecisionContextSource,
) => ResolvedState;

export interface DecisionRunRow {
  fixtureId: string;
  conditionId: string;
  path: DecisionPathName;
  context: DecisionContextSource;
  selected?: string;
  expected: string;
  correct: boolean;
  latencyMs: number;
  generatedTokens: number;
  contextTokens: number;
  /** Probability mass on the option labels. Absent when the path cannot see it. */
  optionMass?: number;
  /** Diagnostic marker only — nothing in the runtime behaves differently. */
  lowOptionMass?: boolean;
  probabilities?: Record<string, number>;
  promptSha256?: string;
  retrievedChunkIds?: string[];
  retrievedStateHit?: boolean;
  error?: string;
  /** True when the readout found no option label at the answer position. */
  unreadableSlot?: boolean;
}

export interface ProbeRunRow {
  probeId: string;
  why: string;
  selected?: string;
  probabilities?: Record<string, number>;
  optionMass?: number;
  lowOptionMass?: boolean;
  latencyMs: number;
  error?: string;
  unreadableSlot?: boolean;
}

export interface Distribution {
  median: number;
  p10: number;
  p90: number;
}

export interface DecisionConditionSummary {
  conditionId: string;
  path: DecisionPathName;
  context: DecisionContextSource;
  runs: number;
  errors: number;
  /** Readouts where no option label appeared at the answer position. */
  unreadableSlots: number;
  correct: number;
  /** Errors count as incorrect: a decision that did not arrive is not accurate. */
  accuracy: number;
  latencyMs: Distribution;
  generatedTokens: Distribution;
  contextTokens: Distribution;
  totalGeneratedTokens: number;
  optionMass?: Distribution;
  lowOptionMassCount: number;
  /** Share of retrieved-context runs whose context held the fixture state. */
  retrievalHitRate?: number;
}

export interface DecisionAgreement {
  context: DecisionContextSource;
  comparedFixtures: number;
  agreed: number;
  agreement: number;
}

export interface DecisionBenchmarkResult {
  fixtureCount: number;
  conditions: Array<{ conditionId: string; path: DecisionPathName; context: DecisionContextSource }>;
  summaries: DecisionConditionSummary[];
  /**
   * How often the two paths picked the same option, per context source.
   *
   * A systems comparison. The direct path reports P(option label | prompt) and
   * the generated path reports a parsed choice; agreement does not make the two
   * readouts semantically equivalent.
   */
  agreements: DecisionAgreement[];
  startedAt: string;
  finishedAt: string;
  durationMs: number;
}

export interface BenchmarkEnvironment {
  userAgent: string;
  platform?: string;
  hardwareConcurrency?: number;
  deviceMemoryGb?: number;
  webgpu: boolean;
  gpu?: {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
  };
  /**
   * `webgpu` for a real run. `stub` marks a run of the measurement apparatus
   * itself against a deterministic fake engine — never a result about a model.
   */
  engine: "webgpu" | "stub";
  timestamp: string;
}

export interface BenchmarkModelRecord {
  modelId: string;
  repo?: string;
  revision?: string;
  pinned: boolean;
  modelLib?: string;
  webllmVersion: string;
  vramRequiredMB?: number;
  contextWindowSize?: number;
}

export interface BenchmarkRuntimeRecord {
  promptVersion: string;
  /** Temperature of the direct readout. Must be 1 for the logprobs to mean anything. */
  readoutTemperature: number;
  /** Temperature of the generated arm, where a deterministic answer is wanted. */
  generationTemperature: number;
  maxTokens: number;
  topLogprobs: number;
  retrievalLimit: number;
  lowOptionMassThreshold: number;
  probabilityStatus: string;
}

/** The exported artifact: one file that can be compared against a later run. */
export interface DecisionBenchmarkArtifact {
  environment: BenchmarkEnvironment;
  model: BenchmarkModelRecord;
  runtime: BenchmarkRuntimeRecord;
  benchmark: DecisionBenchmarkResult;
  results: DecisionRunRow[];
  probes: ProbeRunRow[];
}

export interface DecisionBenchmarkOptions {
  fixtures: DecisionFixture[];
  conditions: DecisionCondition[];
  resolveState: StateResolver;
  onProgress?: (progress: { completed: number; total: number; row: DecisionRunRow }) => void;
}

export interface DecisionBenchmarkRun {
  result: DecisionBenchmarkResult;
  rows: DecisionRunRow[];
}

export async function runDecisionBenchmark(
  options: DecisionBenchmarkOptions,
): Promise<DecisionBenchmarkRun> {
  if (options.conditions.length === 0) {
    throw new Error("The benchmark needs at least one condition.");
  }

  const startedAt = new Date();
  const rows: DecisionRunRow[] = [];
  const total = options.fixtures.length * options.conditions.length;

  for (const fixture of options.fixtures) {
    for (const condition of options.conditions) {
      const row = await runOne(fixture, condition, options.resolveState);
      rows.push(row);
      options.onProgress?.({ completed: rows.length, total, row });
    }
  }

  const finishedAt = new Date();

  return {
    rows,
    result: {
      fixtureCount: options.fixtures.length,
      conditions: options.conditions.map((condition) => ({
        conditionId: conditionId(condition),
        path: condition.path,
        context: condition.context,
      })),
      summaries: options.conditions.map((condition) => summarizeCondition(rows, condition)),
      agreements: measureAgreements(rows),
      startedAt: startedAt.toISOString(),
      finishedAt: finishedAt.toISOString(),
      durationMs: finishedAt.getTime() - startedAt.getTime(),
    },
  };
}

async function runOne(
  fixture: DecisionFixture,
  condition: DecisionCondition,
  resolveState: StateResolver,
): Promise<DecisionRunRow> {
  const base = {
    fixtureId: fixture.id,
    conditionId: conditionId(condition),
    path: condition.path,
    context: condition.context,
    expected: fixture.expected,
  };

  try {
    const resolved = resolveState(fixture, condition.context);
    const result: DecisionResult = await condition.runtime.decide({
      state: resolved.state,
      question: fixture.question,
      options: fixture.options,
    });

    return {
      ...base,
      selected: result.selected,
      correct: result.selected === fixture.expected,
      latencyMs: result.latencyMs,
      generatedTokens: result.trace.generatedTokens,
      contextTokens: result.trace.contextTokens,
      optionMass: result.trace.optionMass,
      lowOptionMass: result.trace.lowOptionMass,
      probabilities: result.probabilities,
      promptSha256: result.trace.promptSha256,
      retrievedChunkIds: resolved.retrievedChunkIds,
      retrievedStateHit: resolved.retrievedStateHit,
    };
  } catch (error) {
    return {
      ...base,
      correct: false,
      latencyMs: 0,
      generatedTokens: 0,
      contextTokens: 0,
      error: error instanceof Error ? error.message : String(error),
      unreadableSlot: isUnreadableSlot(error),
    };
  }
}

export interface OptionMassProbeOptions {
  probes: OptionMassProbe[];
  runtime: DecisionRuntime;
  onProgress?: (progress: { completed: number; total: number; row: ProbeRunRow }) => void;
}

/**
 * Run the probes: decisions whose declared options the model has little reason
 * to support. Nothing here is scored — the point is to watch option mass.
 */
export async function runOptionMassProbes(
  options: OptionMassProbeOptions,
): Promise<ProbeRunRow[]> {
  const rows: ProbeRunRow[] = [];

  for (const probe of options.probes) {
    let row: ProbeRunRow;
    try {
      const result = await options.runtime.decide({
        state: probe.state,
        question: probe.question,
        options: probe.options,
      });
      row = {
        probeId: probe.id,
        why: probe.why,
        selected: result.selected,
        probabilities: result.probabilities,
        optionMass: result.trace.optionMass,
        lowOptionMass: result.trace.lowOptionMass,
        latencyMs: result.latencyMs,
      };
    } catch (error) {
      row = {
        probeId: probe.id,
        why: probe.why,
        latencyMs: 0,
        error: error instanceof Error ? error.message : String(error),
        unreadableSlot: isUnreadableSlot(error),
      };
    }

    rows.push(row);
    options.onProgress?.({ completed: rows.length, total: options.probes.length, row });
  }

  return rows;
}

export function summarizeCondition(
  rows: DecisionRunRow[],
  condition: { path: DecisionPathName; context: DecisionContextSource },
): DecisionConditionSummary {
  const id = conditionId(condition);
  const conditionRows = rows.filter((row) => row.conditionId === id);
  const completed = conditionRows.filter((row) => !row.error);
  const correct = conditionRows.filter((row) => row.correct).length;
  const masses = completed
    .map((row) => row.optionMass)
    .filter((mass): mass is number => typeof mass === "number");
  const retrievalRows = conditionRows.filter((row) => row.retrievedStateHit !== undefined);

  return {
    conditionId: id,
    path: condition.path,
    context: condition.context,
    runs: conditionRows.length,
    errors: conditionRows.length - completed.length,
    unreadableSlots: conditionRows.filter((row) => row.unreadableSlot).length,
    correct,
    accuracy: conditionRows.length > 0 ? correct / conditionRows.length : 0,
    latencyMs: distribution(completed.map((row) => row.latencyMs)),
    generatedTokens: distribution(completed.map((row) => row.generatedTokens)),
    contextTokens: distribution(completed.map((row) => row.contextTokens)),
    totalGeneratedTokens: completed.reduce((sum, row) => sum + row.generatedTokens, 0),
    optionMass: masses.length > 0 ? distribution(masses) : undefined,
    lowOptionMassCount: conditionRows.filter((row) => row.lowOptionMass).length,
    retrievalHitRate:
      retrievalRows.length > 0
        ? retrievalRows.filter((row) => row.retrievedStateHit).length / retrievalRows.length
        : undefined,
  };
}

function measureAgreements(rows: DecisionRunRow[]): DecisionAgreement[] {
  const contexts = [...new Set(rows.map((row) => row.context))];

  return contexts
    .map((context) => {
      const byFixture = new Map<string, Partial<Record<DecisionPathName, string>>>();

      for (const row of rows) {
        if (row.context !== context || !row.selected) {
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
        context,
        comparedFixtures: compared,
        agreed,
        agreement: compared > 0 ? agreed / compared : 0,
      };
    })
    .filter((entry) => entry.comparedFixtures > 0);
}

function isUnreadableSlot(error: unknown): boolean {
  return error instanceof Error && error.name === "DecisionReadoutError";
}

export function median(values: number[]): number {
  return percentile(values, 0.5);
}

export function percentile(values: number[], fraction: number): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  if (sorted.length === 1) {
    return sorted[0]!;
  }

  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);

  if (lower === upper) {
    return sorted[lower]!;
  }

  return sorted[lower]! + (sorted[upper]! - sorted[lower]!) * (position - lower);
}

export function distribution(values: number[]): Distribution {
  return {
    median: median(values),
    p10: percentile(values, 0.1),
    p90: percentile(values, 0.9),
  };
}
