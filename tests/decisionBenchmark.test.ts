import { describe, expect, it } from "vitest";

import decisionFixtures from "../benchmark/decisions.json";
import {
  formatDecisionBenchmarkTable,
  median,
  runDecisionBenchmark,
  type DecisionFixture,
} from "../src/benchmark/DecisionBenchmark";
import {
  DecisionReadoutError,
  type DecisionInput,
  type DecisionResult,
  type DecisionRuntime,
} from "../src/runtime/DecisionRuntime";
import { GeneratedDecisionRuntime, parseSelectedOption } from "../src/runtime/GeneratedDecisionRuntime";
import type { ModelRequest, ModelRuntime } from "../src/runtime/ModelRuntime";

const fixtures = decisionFixtures as DecisionFixture[];

class StubDecisionRuntime implements DecisionRuntime {
  private readonly pick: (input: DecisionInput) => string;
  private readonly generatedTokens: number;

  constructor(pick: (input: DecisionInput) => string, generatedTokens = 0) {
    this.pick = pick;
    this.generatedTokens = generatedTokens;
  }

  async decide(input: DecisionInput): Promise<DecisionResult> {
    const selected = this.pick(input);
    const probabilities = Object.fromEntries(
      input.options.map((option) => [option.id, option.id === selected ? 1 : 0]),
    );

    return {
      selected,
      probabilities,
      latencyMs: this.generatedTokens > 0 ? 400 : 80,
      trace: {
        model: "stub",
        execution: "test",
        readout: this.generatedTokens > 0 ? "generated-text" : "option-logits",
        generatedTokens: this.generatedTokens,
        contextTokens: 100,
        options: input.options.length,
        selected,
        optionMass: 1,
        probabilityStatus: "stub",
        promptVersion: "stub",
        promptSha256: "",
        prompt: "stub",
      },
    };
  }
}

describe("benchmark/decisions.json", () => {
  it("holds 20 well-formed decision fixtures", () => {
    expect(fixtures).toHaveLength(20);
    expect(new Set(fixtures.map((fixture) => fixture.id)).size).toBe(20);

    for (const fixture of fixtures) {
      expect(fixture.state.trim()).not.toBe("");
      expect(fixture.question.trim()).not.toBe("");
      expect(fixture.options.length).toBeGreaterThanOrEqual(2);
      expect(fixture.options.map((option) => option.id)).toContain(fixture.expected);
      expect(new Set(fixture.options.map((option) => option.id)).size).toBe(fixture.options.length);
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

  it("measures both paths over the same fixtures", async () => {
    const report = await runDecisionBenchmark({
      model: "stub-model",
      fixtures: sample,
      paths: {
        generated: new StubDecisionRuntime((input) => input.options[0]!.id, 12),
        direct: new StubDecisionRuntime(
          (input) =>
            sample.find((fixture) => fixture.state === input.state)?.expected ??
            input.options[0]!.id,
        ),
      },
    });

    const generated = report.summaries.find((summary) => summary.path === "generated")!;
    const direct = report.summaries.find((summary) => summary.path === "direct")!;

    expect(report.rows).toHaveLength(sample.length * 2);
    expect(direct.accuracy).toBe(1);
    expect(direct.totalGeneratedTokens).toBe(0);
    expect(generated.totalGeneratedTokens).toBe(sample.length * 12);
    expect(direct.medianLatencyMs).toBeLessThan(generated.medianLatencyMs);
    expect(report.comparedFixtures).toBe(sample.length);
    expect(report.agreement).toBeCloseTo(
      sample.filter((fixture) => fixture.options[0]!.id === fixture.expected).length / sample.length,
    );
  });

  it("counts a failed readout as incorrect instead of dropping the fixture", async () => {
    const report = await runDecisionBenchmark({
      model: "stub-model",
      fixtures: sample,
      paths: {
        direct: new StubDecisionRuntime(() => {
          throw new DecisionReadoutError("no option label in the logits");
        }),
      },
    });

    const direct = report.summaries[0]!;
    expect(direct.runs).toBe(sample.length);
    expect(direct.errors).toBe(sample.length);
    expect(direct.accuracy).toBe(0);
    expect(report.comparedFixtures).toBe(0);
    expect(report.rows.every((row) => row.error?.includes("no option label"))).toBe(true);
  });

  it("renders the Path/Accuracy/Latency/Output-tokens table", async () => {
    const report = await runDecisionBenchmark({
      model: "stub-model",
      fixtures: sample.slice(0, 2),
      paths: { direct: new StubDecisionRuntime((input) => input.options[0]!.id) },
    });

    const table = formatDecisionBenchmarkTable(report);
    expect(table).toContain("| Path | Accuracy | Median latency | Output tokens |");
    expect(table).toContain("| Direct logits |");
    expect(table).toContain("stub-model");
  });
});

describe("median", () => {
  it("averages the middle pair for an even sample", () => {
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([5, 1, 3])).toBe(3);
    expect(median([])).toBe(0);
  });
});

describe("GeneratedDecisionRuntime", () => {
  class ScriptedModel implements ModelRuntime {
    private readonly answer: string;
    constructor(answer: string) {
      this.answer = answer;
    }
    async generate(_request: ModelRequest): Promise<string> {
      return this.answer;
    }
  }

  const input: DecisionInput = {
    state: "The request returned HTTP 401 with the body 'invalid API key'.",
    question: "Should the client retry this request?",
    options: [
      { id: "retry", description: "Retry the request." },
      { id: "stop", description: "Stop retrying and report the failure." },
    ],
  };

  it("parses a letter answer into a typed decision", async () => {
    const runtime = new GeneratedDecisionRuntime(new ScriptedModel("B"), { model: "stub" });
    const result = await runtime.decide(input);

    expect(result.selected).toBe("stop");
    expect(result.probabilities).toEqual({ retry: 0, stop: 1 });
    expect(result.trace.readout).toBe("generated-text");
    expect(result.trace.generatedTokens).toBeGreaterThan(0);
  });

  it("still finds the choice inside a chatty answer", () => {
    expect(parseSelectedOption("The answer is B, because the key is invalid.", input)).toBe("stop");
    expect(parseSelectedOption("stop", input)).toBe("stop");
  });

  it("reports an answer that names no option", () => {
    expect(() => parseSelectedOption("It depends on the situation.", input)).toThrow(
      DecisionReadoutError,
    );
  });
});
