import { describe, expect, it } from "vitest";

import decisionFixtures from "../benchmark/decisions.json";
import probeFixtures from "../benchmark/option-mass-probes.json";
import {
  conditionId,
  distribution,
  median,
  percentile,
  runDecisionBenchmark,
  runOptionMassProbes,
  type DecisionContextSource,
  type DecisionFixture,
  type OptionMassProbe,
  type StateResolver,
} from "../src/benchmark/DecisionBenchmark";
import {
  buildBenchmarkArtifact,
  formatDecisionBenchmarkReport,
  formatDecisionBenchmarkTable,
} from "../src/benchmark/DecisionReport";
import {
  DecisionReadoutError,
  type DecisionInput,
  type DecisionResult,
  type DecisionRuntime,
} from "../src/runtime/DecisionRuntime";

const fixtures = decisionFixtures as DecisionFixture[];
const probes = probeFixtures as OptionMassProbe[];

/** Answers from the state text, so context quality changes the outcome. */
class StubDecisionRuntime implements DecisionRuntime {
  private readonly readout: "option-logits" | "generated-text";
  private readonly generatedTokens: number;
  private readonly optionMass?: number;
  private readonly pick?: (input: DecisionInput) => string;

  constructor(options: {
    readout: "option-logits" | "generated-text";
    generatedTokens?: number;
    optionMass?: number;
    pick?: (input: DecisionInput) => string;
  }) {
    this.readout = options.readout;
    this.generatedTokens = options.generatedTokens ?? 0;
    this.optionMass = options.optionMass;
    this.pick = options.pick;
  }

  async decide(input: DecisionInput): Promise<DecisionResult> {
    const selected =
      this.pick?.(input) ??
      fixtures.find((fixture) => input.state.includes(fixture.state))?.expected ??
      // No state, no evidence: fall back to the last option, which is never the
      // expected answer in the fixtures this stub is used with.
      input.options[input.options.length - 1]!.id;

    return {
      selected,
      probabilities: Object.fromEntries(
        input.options.map((option) => [option.id, option.id === selected ? 1 : 0]),
      ),
      latencyMs: this.generatedTokens > 0 ? 400 : 80,
      trace: {
        model: "stub",
        execution: "test",
        readout: this.readout,
        generatedTokens: this.generatedTokens,
        contextTokens: Math.ceil(input.state.length / 4),
        options: input.options.length,
        selected,
        optionMass: this.optionMass,
        lowOptionMass: this.optionMass === undefined ? undefined : this.optionMass < 0.5,
        probabilityStatus: "stub",
        promptVersion: "stub",
        promptSha256: "0".repeat(64),
        prompt: "stub",
      },
    };
  }
}

/** Stands in for the browser resolver, which needs a DOM. */
const resolveState: StateResolver = (fixture, context: DecisionContextSource) => {
  if (context === "state") {
    return { state: fixture.state };
  }

  if (context === "full-page") {
    return { state: stripTags(fixture.pageHtml) };
  }

  // "Retrieval" that finds the state for every fixture but the first, so the
  // hit-rate plumbing is exercised in both directions.
  const hit = fixture.id !== fixtures[0]!.id;
  return {
    state: hit ? fixture.state : "unrelated passage about shipping times",
    retrievedChunkIds: hit ? ["chunk-3"] : ["chunk-1"],
    retrievedStateHit: hit,
  };
};

function stripTags(html: string): string {
  return html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}

describe("benchmark/decisions.json", () => {
  it("holds 20 well-formed decision fixtures", () => {
    expect(fixtures).toHaveLength(20);
    expect(new Set(fixtures.map((fixture) => fixture.id)).size).toBe(20);

    for (const fixture of fixtures) {
      expect(fixture.options.map((option) => option.id)).toContain(fixture.expected);
      expect(new Set(fixture.options.map((option) => option.id)).size).toBe(fixture.options.length);
      expect(fixture.query.trim()).not.toBe("");
    }
  });

  it("gives every fixture a page that contains its state among distractors", () => {
    for (const fixture of fixtures) {
      expect(fixture.pageHtml).toContain(fixture.state);
      // The state must not be the only thing on the page, or full-page and
      // retrieved context would be the same measurement.
      const pageText = stripTags(fixture.pageHtml);
      expect(pageText.length).toBeGreaterThan(fixture.state.length * 2);
      expect((fixture.pageHtml.match(/<h2>/g) ?? []).length).toBeGreaterThanOrEqual(4);
    }
  });

  it("starts binary and adds a few 3-4 option decisions", () => {
    const binary = fixtures.filter((fixture) => fixture.options.length === 2);
    const wider = fixtures.filter((fixture) => fixture.options.length > 2);

    expect(binary.length).toBeGreaterThanOrEqual(15);
    expect(wider.length).toBeGreaterThanOrEqual(3);
    expect(Math.max(...wider.map((fixture) => fixture.options.length))).toBeLessThanOrEqual(4);
  });
});

describe("runDecisionBenchmark", () => {
  const sample = fixtures.slice(0, 4);

  const conditions = (): Array<{
    path: "generated" | "direct";
    context: DecisionContextSource;
    runtime: DecisionRuntime;
  }> => [
    {
      path: "generated",
      context: "full-page",
      runtime: new StubDecisionRuntime({ readout: "generated-text", generatedTokens: 12 }),
    },
    {
      path: "generated",
      context: "retrieved",
      runtime: new StubDecisionRuntime({ readout: "generated-text", generatedTokens: 12 }),
    },
    {
      path: "direct",
      context: "full-page",
      runtime: new StubDecisionRuntime({ readout: "option-logits", optionMass: 0.8 }),
    },
    {
      path: "direct",
      context: "retrieved",
      runtime: new StubDecisionRuntime({ readout: "option-logits", optionMass: 0.8 }),
    },
  ];

  it("runs the full context matrix over the same fixtures", async () => {
    const { result, rows } = await runDecisionBenchmark({
      fixtures: sample,
      conditions: conditions(),
      resolveState,
    });

    expect(rows).toHaveLength(sample.length * 4);
    expect(result.summaries.map((summary) => summary.conditionId)).toEqual([
      "generated/full-page",
      "generated/retrieved",
      "direct/full-page",
      "direct/retrieved",
    ]);

    const directRetrieved = result.summaries.find(
      (summary) => summary.conditionId === "direct/retrieved",
    )!;
    // One fixture's retrieval misses the state on purpose.
    expect(directRetrieved.retrievalHitRate).toBeCloseTo(0.75);
    expect(directRetrieved.correct).toBe(3);
    expect(directRetrieved.optionMass?.median).toBeCloseTo(0.8);
    expect(directRetrieved.totalGeneratedTokens).toBe(0);

    const generatedRetrieved = result.summaries.find(
      (summary) => summary.conditionId === "generated/retrieved",
    )!;
    expect(generatedRetrieved.totalGeneratedTokens).toBe(sample.length * 12);
    expect(generatedRetrieved.optionMass).toBeUndefined();
  });

  it("reports agreement separately for each context source", async () => {
    const { result } = await runDecisionBenchmark({
      fixtures: sample,
      conditions: [
        ...conditions().slice(0, 3),
        {
          path: "direct",
          context: "retrieved",
          // Disagrees with the generated path on every retrieved-context row.
          runtime: new StubDecisionRuntime({
            readout: "option-logits",
            optionMass: 0.8,
            pick: (input) => input.options[input.options.length - 1]!.id,
          }),
        },
      ],
      resolveState,
    });

    const byContext = Object.fromEntries(
      result.agreements.map((entry) => [entry.context, entry.agreement]),
    );

    expect(byContext["full-page"]).toBe(1);
    expect(byContext["retrieved"]).toBeLessThan(1);
  });

  it("counts a failed readout as incorrect and marks the unreadable slot", async () => {
    const { result, rows } = await runDecisionBenchmark({
      fixtures: sample,
      conditions: [
        {
          path: "direct",
          context: "state",
          runtime: new StubDecisionRuntime({
            readout: "option-logits",
            pick: () => {
              throw new DecisionReadoutError("no option label in the logits");
            },
          }),
        },
      ],
      resolveState,
    });

    const summary = result.summaries[0]!;
    expect(summary.runs).toBe(sample.length);
    expect(summary.errors).toBe(sample.length);
    expect(summary.unreadableSlots).toBe(sample.length);
    expect(summary.accuracy).toBe(0);
    expect(rows.every((row) => row.unreadableSlot)).toBe(true);
    expect(result.agreements).toEqual([]);
  });

  it("rejects a run with no conditions", async () => {
    await expect(
      runDecisionBenchmark({ fixtures: sample, conditions: [], resolveState }),
    ).rejects.toThrow("at least one condition");
  });
});

describe("runOptionMassProbes", () => {
  it("records mass without scoring the answer", async () => {
    const rows = await runOptionMassProbes({
      probes: probes.slice(0, 3),
      runtime: new StubDecisionRuntime({
        readout: "option-logits",
        optionMass: 0.18,
        pick: (input) => input.options[0]!.id,
      }),
    });

    expect(rows).toHaveLength(3);
    for (const row of rows) {
      expect(row.optionMass).toBeCloseTo(0.18);
      expect(row.lowOptionMass).toBe(true);
      expect(row.why).not.toBe("");
      expect(row).not.toHaveProperty("correct");
    }
  });
});

describe("the exported artifact", () => {
  it("carries the model pin, prompt version and environment with the results", async () => {
    const { result, rows } = await runDecisionBenchmark({
      fixtures: fixtures.slice(0, 2),
      conditions: [
        {
          path: "direct",
          context: "retrieved",
          runtime: new StubDecisionRuntime({ readout: "option-logits", optionMass: 0.94 }),
        },
        {
          path: "generated",
          context: "retrieved",
          runtime: new StubDecisionRuntime({ readout: "generated-text", generatedTokens: 1 }),
        },
      ],
      resolveState,
    });

    const artifact = buildBenchmarkArtifact({
      environment: {
        userAgent: "test-agent",
        webgpu: true,
        engine: "webgpu",
        gpu: { vendor: "test-vendor", architecture: "test-arch" },
        timestamp: "2026-09-17T00:00:00.000Z",
      },
      model: {
        modelId: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
        repo: "mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
        revision: "a".repeat(40),
        pinned: true,
        webllmVersion: "0.2.85",
      },
      runtime: {
        promptVersion: "browser-direct-options-v1",
        readoutTemperature: 1,
        generationTemperature: 0,
        maxTokens: 1,
        topLogprobs: 5,
        retrievalLimit: 5,
        lowOptionMassThreshold: 0.5,
        probabilityStatus: "conditional option score",
      },
      benchmark: result,
      results: rows,
      probes: [],
    });

    expect(Object.keys(artifact)).toEqual([
      "environment",
      "model",
      "runtime",
      "benchmark",
      "results",
      "probes",
    ]);
    expect(artifact.results[0]!.promptSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(JSON.parse(JSON.stringify(artifact))).toEqual(artifact);

    const report = formatDecisionBenchmarkReport(artifact);
    expect(report).toContain("Browser Decision Benchmark");
    expect(report).toContain("aaaaaaaaaa");
    expect(report).toContain("Accuracy matrix");
    expect(report).toContain("Option mass (direct)");
    expect(report).not.toContain("UNPINNED");
    // The report never calls the direct probabilities confidence.
    expect(report.toLowerCase()).not.toContain("confidence");

    expect(formatDecisionBenchmarkTable(artifact)).toContain(
      "| Context | Path | Accuracy | Median latency | Output tokens | Option mass (median) |",
    );
  });

  it("flags an unpinned revision in the report", async () => {
    const { result, rows } = await runDecisionBenchmark({
      fixtures: fixtures.slice(0, 1),
      conditions: [
        {
          path: "direct",
          context: "state",
          runtime: new StubDecisionRuntime({ readout: "option-logits", optionMass: 0.9 }),
        },
      ],
      resolveState,
    });

    const report = formatDecisionBenchmarkReport(
      buildBenchmarkArtifact({
        environment: {
          userAgent: "test-agent",
          webgpu: false,
          engine: "stub",
          timestamp: "2026-09-17T00:00:00.000Z",
        },
        model: { modelId: "m", revision: "main", pinned: false, webllmVersion: "0.2.85" },
        runtime: {
          promptVersion: "v1",
          readoutTemperature: 1,
          generationTemperature: 0,
          maxTokens: 1,
          topLogprobs: 5,
          retrievalLimit: 5,
          lowOptionMassThreshold: 0.5,
          probabilityStatus: "stub",
        },
        benchmark: result,
        results: rows,
        probes: [],
      }),
    );

    expect(report).toContain("UNPINNED");
    expect(report).toContain("STUB ENGINE (not a model result)");
  });
});

describe("statistics", () => {
  it("reports median and tails", () => {
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([5, 1, 3])).toBe(3);
    expect(median([])).toBe(0);
    expect(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1)).toBeCloseTo(1.9);
    expect(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9)).toBeCloseTo(9.1);
    expect(distribution([1, 1, 1])).toEqual({ median: 1, p10: 1, p90: 1 });
  });
});

describe("conditionId", () => {
  it("names a cell of the matrix", () => {
    expect(conditionId({ path: "direct", context: "retrieved" })).toBe("direct/retrieved");
  });
});
