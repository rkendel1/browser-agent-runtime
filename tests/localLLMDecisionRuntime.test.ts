import { describe, expect, it } from "vitest";

import {
  DecisionBatch,
  MAX_DECISION_OPTIONS,
  encodeCandidateLabels,
  type DecisionInput,
} from "../src/runtime/DecisionRuntime";
import {
  LocalLLMDecisionRuntime,
} from "../src/runtime/LocalLLMDecisionRuntime";
import type { LocalLLMAdapter } from "../src/runtime/LocalLLMAdapter";

const adapter: LocalLLMAdapter = {
  capabilities: {
    binary: true,
    choice: true,
    score: true,
    batch: true,
    distribution: true,
    executionMethods: ["logit"],
  },
  execute: async ({ candidateLabels }) => ({
    candidates: candidateLabels.map((token, index) => ({
      token,
      logprob: Math.log(candidateLabels.length - index),
    })),
    executionMethod: "logit",
    model: "fixture-model",
    modelRevision: "unavailable",
  }),
};

describe("candidate-label encoding", () => {
  it.each([2, 3, 4, MAX_DECISION_OPTIONS])("encodes %s options deterministically", (count) => {
    const options = Array.from({ length: count }, (_, index) => `option-${index}`);
    expect([...encodeCandidateLabels(options).entries()]).toEqual(
      options.map((option, index) => [option, String.fromCharCode(65 + index)]),
    );
  });

  it("rejects duplicate and invalid option sets", () => {
    expect(() => encodeCandidateLabels(["same", "same"])).toThrow("Duplicate");
    expect(() => encodeCandidateLabels(["only"])).toThrow("between 2");
    expect(() =>
      encodeCandidateLabels(Array.from({ length: MAX_DECISION_OPTIONS + 1 }, (_, i) => String(i))),
    ).toThrow("between 2");
  });
});

describe("LocalLLMDecisionRuntime", () => {
  it("supports binary, choice, score, provenance, and independent batches", async () => {
    const runtime = new LocalLLMDecisionRuntime(adapter, {
      runtime: "browser-agent-runtime",
      runtimeVersion: "test",
    });
    const input: DecisionInput = {
      state: "An invoice is overdue.",
      question: "Escalate it?",
      options: [
        { id: "yes", description: "Yes" },
        { id: "no", description: "No" },
      ],
    };

    const binary = await runtime.decideBinary({ state: input.state, question: input.question });
    expect(typeof binary.selected).toBe("boolean");
    expect(binary.trace.provenance).toMatchObject({
      runtime: "browser-agent-runtime",
      executionMethod: "logit",
      modelRevision: "unavailable",
    });

    const choice = await runtime.decideChoice({
      state: input.state,
      question: input.question,
      options: ["yes", "no"],
    });
    expect(["yes", "no"]).toContain(choice.selected);

    const score = await runtime.decideScore({
      state: input.state,
      question: "How urgent is it?",
      options: [0, 1, 2],
    });
    expect([0, 1, 2]).toContain(score.selected);

    const batch = await new DecisionBatch(runtime).decide([input, { ...input, question: "Review it?" }]);
    expect(batch).toHaveLength(2);
    expect(batch[0]!.trace.prompt).not.toBe(batch[1]!.trace.prompt);
  });

  it("does not report unsupported option mass as zero", async () => {
    const unsupported: LocalLLMAdapter = {
      ...adapter,
      execute: async () => ({
        candidates: [
          { token: "A", logprob: Math.log(0.8) },
          { token: "B", logprob: Math.log(0.2) },
        ],
        executionMethod: "logit",
        optionMassStatus: "unsupported",
      }),
    };
    const result = await new LocalLLMDecisionRuntime(unsupported).decide({
      state: "state",
      question: "question",
      options: [
        { id: "a", description: "A" },
        { id: "b", description: "B" },
      ],
    });
    expect(result.trace.optionMassStatus).toBe("unsupported");
    expect(result.trace.optionMass).toBeUndefined();
    expect(result.trace.lowOptionMass).toBeUndefined();
  });
});
