import { describe, expect, it } from "vitest";

import {
  decisionsConform,
  runBrowserAgentBenchmark,
  type BrowserBenchmarkMetadata,
} from "../src/benchmark/BrowserAgentBenchmark";
import type { BrowserAgentRuntime, BrowserRunResult } from "../src/browser/BrowserTypes";

const metadata: BrowserBenchmarkMetadata = {
  modelId: "fixture",
  modelRevision: "main",
  runtimeVersion: "1.0.0",
  webllmVersion: "0.2.85",
  browser: "fixture-browser",
  gpu: "fixture-gpu",
  promptVersion: "v1",
  contextMode: "state",
};

function result(optionId: string): BrowserRunResult {
  return {
    status: "complete",
    steps: 1,
    latencyMs: 9,
    evidence: [
      {
        snapshotId: "a",
        decision: {
          optionId,
          path: "direct",
          latencyMs: 3,
          contextTokens: 12,
          generatedTokens: 0,
        },
        action: { type: "stop" },
        execution: { status: "success", durationMs: 2 },
      },
    ],
  };
}

function runtimeReturning(runResult: BrowserRunResult): BrowserAgentRuntime {
  return {
    run: async () => runResult,
    observe: async () => { throw new Error("unused"); },
    decide: async () => { throw new Error("unused"); },
    execute: async () => { throw new Error("unused"); },
  };
}

describe("browser agent benchmark", () => {
  it("records metrics, probes, and unpinned reproducibility honestly", async () => {
    const runtime = runtimeReturning(result("stop"));
    const benchmark = await runBrowserAgentBenchmark(
      runtime,
      [
        {
          id: "empty",
          task: { instruction: "Stop", options: [] },
          expectedOptionIds: ["stop"],
          probe: "empty-state",
        },
      ],
      metadata,
    );
    expect(benchmark).toMatchObject({
      accuracy: 1,
      metadata: { reproducible: false, reproducibility: "UNPINNED — not reproducible" },
      totals: { decisionLatencyMs: 3, executionLatencyMs: 2, totalLatencyMs: 9, steps: 1 },
      probes: { "empty-state": { runs: 1, passed: 1, rate: 1 } },
    });
  });

  it("marks a commit revision reproducible", async () => {
    const runtime = runtimeReturning(result("stop"));
    const benchmark = await runBrowserAgentBenchmark(runtime, [], {
      ...metadata,
      modelRevision: "a".repeat(40),
    });
    expect(benchmark.metadata).toMatchObject({ reproducible: true, reproducibility: "PINNED" });
  });

  it("checks direct/generated canonical conformance by option id", () => {
    const direct = result("continue");
    const generated = structuredClone(direct);
    generated.evidence[0]!.decision.path = "generated";
    expect(decisionsConform(direct, generated)).toBe(true);
    generated.evidence[0]!.decision.optionId = "stop";
    expect(decisionsConform(direct, generated)).toBe(false);
  });
});
