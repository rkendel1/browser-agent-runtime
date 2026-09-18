import { describe, expect, it } from "vitest";

import { validateDecision } from "../src/browser/BrowserValidation";
import type {
  BrowserDecision,
  BrowserDecisionRequest,
  BrowserExecutor,
  BrowserObservation,
} from "../src/browser/BrowserTypes";

function observation(snapshotId = "a"): BrowserObservation {
  return {
    url: "https://example.test/page",
    viewport: { width: 800, height: 600 },
    elements: [
      { id: "button", role: "button", visible: true, enabled: true },
      { id: "input", role: "input", visible: true, enabled: true },
      { id: "disabled", role: "button", visible: true, enabled: false, disabled: true },
    ],
    timestamp: 1,
    provenance: { source: "browser", snapshotId },
  };
}
const executor: BrowserExecutor = {
  observe: async () => observation(),
  click: async () => undefined,
  type: async () => undefined,
  navigate: async () => undefined,
};
function setup(action: BrowserDecisionRequest["options"][number]["action"]) {
  const source = observation();
  const request: BrowserDecisionRequest = {
    observation: source,
    instruction: "test",
    options: [
      { id: "chosen", description: "Chosen", action },
      { id: "stop", description: "Stop", action: { type: "stop" } },
    ],
  };
  const decision: BrowserDecision = {
    optionId: "chosen",
    confidenceStatus: "unavailable",
    optionMassStatus: "unavailable",
    path: "direct",
    model: { id: "fixture", runtime: "fixture" },
    evidence: { snapshotId: "a" },
    latencyMs: 1,
  };
  return { source, request, decision };
}

describe("browser decision validation", () => {
  it("accepts a valid click", () => {
    const { request, decision } = setup({ type: "click", elementId: "button" });
    expect(validateDecision(decision, request, observation(), executor)).toMatchObject({ ok: true });
  });

  it("rejects stale snapshots", () => {
    const { request, decision } = setup({ type: "click", elementId: "button" });
    expect(validateDecision(decision, request, observation("b"), executor)).toMatchObject({
      ok: false,
      reason: "stale_decision",
    });
  });

  it.each([
    [{ type: "click", elementId: "missing" } as const, "missing_element"],
    [{ type: "click", elementId: "disabled" } as const, "element_disabled"],
    [{ type: "type", elementId: "button", value: "x" } as const, "invalid_role_action"],
  ])("blocks invalid element action %#", (action, reason) => {
    const { request, decision } = setup(action);
    expect(validateDecision(decision, request, observation(), executor)).toMatchObject({ ok: false, reason });
  });

  it("rejects cross-origin and script navigation by default", () => {
    for (const url of ["https://attacker.test/", "javascript:alert(1)"]) {
      const { request, decision } = setup({ type: "navigate", url });
      expect(validateDecision(decision, request, observation(), executor)).toMatchObject({
        ok: false,
        reason: "invalid_navigation",
      });
    }
  });

  it("rejects a capability the executor does not have", () => {
    const { request, decision } = setup({ type: "select", elementId: "input", value: "x" });
    expect(validateDecision(decision, request, observation(), executor)).toMatchObject({
      ok: false,
      reason: "missing_capability",
    });
  });
});
