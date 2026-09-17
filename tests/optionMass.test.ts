import { describe, expect, it } from "vitest";

import probeFixtures from "../benchmark/option-mass-probes.json";
import {
  LOW_OPTION_MASS_THRESHOLD,
  resolveOptionProbabilities,
  type DecisionOption,
} from "../src/runtime/DecisionRuntime";
import {
  WebLLMDecisionRuntime,
  type WebLLMDecisionEngine,
} from "../src/runtime/WebLLMDecisionRuntime";
import type { OptionMassProbe } from "../src/benchmark/DecisionBenchmark";

const probes = probeFixtures as OptionMassProbe[];

const options: DecisionOption[] = [
  { id: "yes", description: "The claim holds." },
  { id: "no", description: "The claim does not hold." },
];

function engineWithSlot(
  topLogprobs: Array<{ token: string; logprob: number }>,
): WebLLMDecisionEngine {
  return {
    reload: async () => undefined,
    chat: {
      completions: {
        create: async () => ({
          choices: [
            {
              logprobs: {
                content: [
                  {
                    token: topLogprobs[0]!.token,
                    logprob: topLogprobs[0]!.logprob,
                    top_logprobs: topLogprobs,
                  },
                ],
              },
            },
          ],
        }),
      },
    },
  };
}

describe("option mass", () => {
  it("separates a near-tie from a model that was not answering", () => {
    // 51/49 between the labels, but the labels only held 18% of the mass.
    const { selected, probabilities, optionMass } = resolveOptionProbabilities(options, [
      { token: "I", logprob: Math.log(0.62) },
      { token: "It", logprob: Math.log(0.2) },
      { token: "A", logprob: Math.log(0.0918) },
      { token: "B", logprob: Math.log(0.0882) },
    ]);

    expect(selected).toBe("yes");
    expect(probabilities.yes).toBeCloseTo(0.51, 2);
    expect(probabilities.no).toBeCloseTo(0.49, 2);
    // The renormalized numbers say 51%. The mass says most of the model's
    // next-token probability went somewhere else entirely.
    expect(optionMass).toBeCloseTo(0.18, 2);
  });

  it("marks low mass as a diagnostic without changing the decision", async () => {
    const runtime = new WebLLMDecisionRuntime({
      createEngine: async () =>
        engineWithSlot([
          { token: "The", logprob: Math.log(0.82) },
          { token: "A", logprob: Math.log(0.0918) },
          { token: "B", logprob: Math.log(0.0882) },
        ]),
    });

    const result = await runtime.decide({
      state: "No data was retrieved for this request.",
      question: "Does the claim hold?",
      options,
    });

    expect(result.selected).toBe("yes");
    expect(result.trace.optionMass).toBeLessThan(LOW_OPTION_MASS_THRESHOLD);
    expect(result.trace.lowOptionMass).toBe(true);
    // The probabilities are reported as they are; the flag is a marker, not a
    // policy that suppresses or rewrites the decision.
    expect(result.probabilities.yes).toBeCloseTo(0.51, 2);
    expect(result.trace.probabilityStatus).toContain("uncalibrated");
  });

  it("does not mark a decision the model actually committed to", async () => {
    const runtime = new WebLLMDecisionRuntime({
      createEngine: async () =>
        engineWithSlot([
          { token: "A", logprob: Math.log(0.93) },
          { token: "B", logprob: Math.log(0.05) },
          { token: "The", logprob: Math.log(0.02) },
        ]),
    });

    const result = await runtime.decide({
      state: "Health checks passed in all three zones.",
      question: "Does the claim hold?",
      options,
    });

    expect(result.trace.optionMass).toBeCloseTo(0.98, 2);
    expect(result.trace.lowOptionMass).toBe(false);
  });

  it("counts every spelling of a slot toward the mass", () => {
    const { optionMass } = resolveOptionProbabilities(options, [
      { token: " A", logprob: Math.log(0.4) },
      { token: "A", logprob: Math.log(0.3) },
      { token: "B", logprob: Math.log(0.1) },
      { token: "Well", logprob: Math.log(0.2) },
    ]);

    expect(optionMass).toBeCloseTo(0.8, 6);
  });
});

describe("benchmark/option-mass-probes.json", () => {
  it("holds probes with no correct answer and a stated reason", () => {
    expect(probes.length).toBeGreaterThanOrEqual(5);
    expect(new Set(probes.map((probe) => probe.id)).size).toBe(probes.length);

    for (const probe of probes) {
      expect(probe).not.toHaveProperty("expected");
      expect(probe.why.trim()).not.toBe("");
      expect(probe.options.length).toBeGreaterThanOrEqual(2);
      expect(probe.state.trim()).not.toBe("");
    }
  });
});
