import { describe, expect, it, vi } from "vitest";

import {
  DECISION_PROMPT_VERSION,
  DecisionReadoutError,
  buildDecisionPrompt,
  resolveOptionProbabilities,
  type DecisionInput,
} from "../src/runtime/DecisionRuntime";
import {
  WebLLMDecisionRuntime,
  type WebLLMDecisionEngine,
} from "../src/runtime/WebLLMDecisionRuntime";

const input: DecisionInput = {
  state: "Customer cannot access their account after a password reset.",
  question: "Which queue should handle this request?",
  options: [
    { id: "access", description: "Account access support." },
    { id: "billing", description: "Billing support." },
  ],
};

function engineWithLogprobs(
  topLogprobs: Array<{ token: string; logprob: number }>,
  usagePromptTokens = 96,
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
                    token: topLogprobs[0]?.token ?? "",
                    logprob: topLogprobs[0]?.logprob ?? 0,
                    top_logprobs: topLogprobs,
                  },
                ],
              },
            },
          ],
          usage: { prompt_tokens: usagePromptTokens },
        }),
      },
    },
  };
}

describe("buildDecisionPrompt", () => {
  it("lays the options out as labelled answer slots", () => {
    expect(buildDecisionPrompt(input)).toBe(
      [
        "State:",
        "Customer cannot access their account after a password reset.",
        "Question:",
        "Which queue should handle this request?",
        "Options:",
        "A. Account access support.",
        "B. Billing support.",
        "Select the option that best answers the question.",
      ].join("\n"),
    );
  });

  it("is deterministic for the same decision", () => {
    expect(buildDecisionPrompt(input)).toBe(buildDecisionPrompt(structuredClone(input)));
  });
});

describe("resolveOptionProbabilities", () => {
  it("normalizes option-label logits into a distribution", () => {
    const { selected, probabilities, optionMass } = resolveOptionProbabilities(input.options, [
      { token: "A", logprob: Math.log(0.9) },
      { token: "B", logprob: Math.log(0.09) },
      { token: "The", logprob: Math.log(0.01) },
    ]);

    expect(selected).toBe("access");
    expect(probabilities.access).toBeCloseTo(0.909, 2);
    expect(probabilities.billing).toBeCloseTo(0.091, 2);
    expect(probabilities.access! + probabilities.billing!).toBeCloseTo(1, 10);
    // Only the two label tokens count toward the allowed-token mass.
    expect(optionMass).toBeCloseTo(0.99, 6);
  });

  it("gives an unobserved option probability zero without dropping it", () => {
    const { probabilities } = resolveOptionProbabilities(input.options, [
      { token: "A", logprob: Math.log(0.8) },
    ]);

    expect(probabilities).toEqual({ access: 1, billing: 0 });
  });

  it("reads the same option through different token spellings", () => {
    const { selected, optionMass } = resolveOptionProbabilities(input.options, [
      { token: " B", logprob: Math.log(0.5) },
      { token: "B.", logprob: Math.log(0.3) },
      { token: "A", logprob: Math.log(0.2) },
    ]);

    expect(selected).toBe("billing");
    expect(optionMass).toBeCloseTo(1, 6);
  });

  it("ignores words that merely start with a label letter", () => {
    const { selected } = resolveOptionProbabilities(input.options, [
      { token: "Billing", logprob: Math.log(0.6) },
      { token: "A", logprob: Math.log(0.3) },
    ]);

    expect(selected).toBe("access");
  });

  it("refuses to guess when no option reached the answer slot", () => {
    expect(() =>
      resolveOptionProbabilities(input.options, [
        { token: "I", logprob: Math.log(0.7) },
        { token: "Sure", logprob: Math.log(0.2) },
      ]),
    ).toThrow(DecisionReadoutError);
  });
});

describe("WebLLMDecisionRuntime", () => {
  it("decides from the answer-slot logits without generating an answer", async () => {
    const engine = engineWithLogprobs([
      { token: "A", logprob: Math.log(0.91) },
      { token: "B", logprob: Math.log(0.09) },
    ]);
    const create = vi.spyOn(engine.chat.completions, "create");
    const runtime = new WebLLMDecisionRuntime({
      model: "test-model",
      createEngine: async () => engine,
    });

    const result = await runtime.decide(input);

    expect(result.selected).toBe("access");
    expect(result.probabilities.access).toBeGreaterThan(result.probabilities.billing!);
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
    expect(result.trace).toMatchObject({
      model: "test-model",
      readout: "option-logits",
      generatedTokens: 0,
      contextTokens: 96,
      options: 2,
      selected: "access",
      promptVersion: DECISION_PROMPT_VERSION,
    });
    expect(result.trace.promptSha256).toMatch(/^[0-9a-f]{64}$/);

    const request = create.mock.calls[0]![0];
    expect(request.max_tokens).toBe(1);
    expect(request.temperature).toBe(0);
    expect(request.logprobs).toBe(true);
    expect(request.top_logprobs).toBe(5);
    expect(request.messages[1]!.content).toBe(buildDecisionPrompt(input));
  });

  it("caps top_logprobs at what the engine supports", async () => {
    const engine = engineWithLogprobs([{ token: "A", logprob: Math.log(0.99) }]);
    const create = vi.spyOn(engine.chat.completions, "create");
    const runtime = new WebLLMDecisionRuntime({
      topLogprobs: 20,
      createEngine: async () => engine,
    });

    await runtime.decide(input);

    expect(create.mock.calls[0]![0].top_logprobs).toBe(5);
  });

  it("rejects a decision with fewer than two options", async () => {
    const runtime = new WebLLMDecisionRuntime({
      createEngine: async () => engineWithLogprobs([{ token: "A", logprob: 0 }]),
    });

    await expect(
      runtime.decide({ ...input, options: [{ id: "only", description: "Only option." }] }),
    ).rejects.toThrow("at least two options");
  });

  it("fails loudly when the engine reports no logprobs", async () => {
    const runtime = new WebLLMDecisionRuntime({
      createEngine: async () => ({
        reload: async () => undefined,
        chat: { completions: { create: async () => ({ choices: [{}] }) } },
      }),
    });

    await expect(runtime.decide(input)).rejects.toThrow("no logprobs");
  });
});
