import { describe, expect, it } from "vitest";

import {
  browserDecisionOption,
  deterministicBrowserOptionId,
  normalizeDirectDecision,
  normalizeGeneratedDecision,
} from "../src/browser/BrowserDecision";
import type { BrowserDecisionRequest, BrowserObservation } from "../src/browser/BrowserTypes";
import type { DecisionResult } from "../src/runtime/DecisionRuntime";

const observation: BrowserObservation = {
  url: "https://example.test/",
  viewport: { width: 800, height: 600 },
  elements: [],
  timestamp: 1,
  provenance: { source: "browser", snapshotId: "snapshot-a" },
};
const options = [
  browserDecisionOption("Continue", { type: "click", elementId: "continue" }, "continue"),
  browserDecisionOption("Stop", { type: "stop" }, "stop"),
];
const request: BrowserDecisionRequest = { observation, instruction: "Continue if ready", options };

function result(readout: "option-logits" | "generated-text", selected = "continue"): DecisionResult {
  return {
    selected,
    probabilities: { continue: 0.8, stop: 0.2 },
    latencyMs: 12,
    trace: {
      model: "fixture",
      execution: "test",
      readout,
      generatedTokens: readout === "generated-text" ? 1 : 0,
      contextTokens: 20,
      options: 2,
      selected,
      optionMass: readout === "option-logits" ? 0.7 : undefined,
      optionMassStatus: readout === "option-logits" ? "available" : "unsupported",
      probabilityStatus: "test",
      promptVersion: "v1",
      promptSha256: "abc",
      prompt: "prompt",
      provenance: {
        runtime: "fixture-runtime",
        runtimeVersion: "1",
        model: "fixture",
        modelRevision: "abc123",
        executionMethod: readout === "option-logits" ? "logit" : "generated",
        promptRevision: "v1",
        decisionSchemaRevision: "v1",
      },
    },
  };
}

describe("browser decision normalization", () => {
  it("normalizes a direct decision", () => {
    expect(normalizeDirectDecision(result("option-logits"), request)).toMatchObject({
      optionId: "continue",
      path: "direct",
      confidence: 0.8,
      confidenceStatus: "available",
      optionMass: 0.7,
      optionMassStatus: "available",
      evidence: { snapshotId: "snapshot-a", generatedTokens: 0 },
    });
  });

  it("normalizes generated output without claiming probability diagnostics", () => {
    expect(normalizeGeneratedDecision(result("generated-text"), request)).toMatchObject({
      optionId: "continue",
      path: "generated",
      confidenceStatus: "unsupported",
      optionMassStatus: "unsupported",
    });
  });

  it("rejects an unknown option", () => {
    expect(() => normalizeDirectDecision(result("option-logits", "invented"), request)).toThrow(
      "unknown option",
    );
  });

  it("creates deterministic ids from action content independent of key order", () => {
    const left = deterministicBrowserOptionId({ type: "type", elementId: "name", value: "Randy" });
    const right = deterministicBrowserOptionId({ value: "Randy", elementId: "name", type: "type" });
    expect(left).toBe(right);
  });
});
