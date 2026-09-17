import { describe, expect, it, vi } from "vitest";

import {
  DecisionReadoutError,
  buildDecisionPrompt,
  resolveOptionProbabilities,
  type DecisionInput,
} from "../src/runtime/DecisionRuntime";
import {
  WebLLMDecisionRuntime,
  type WebLLMDecisionEngine,
} from "../src/runtime/WebLLMDecisionRuntime";

/**
 * The contract under test:
 *
 *   slot token → option id
 *
 * and never:
 *
 *   generated text → parsed option
 *
 * The answer slot is the letter. An option's wording must not be able to select
 * it, however tempting the semantics are — "Yes", "Approve" and "Escalate" are
 * words the model might emit, not answer slots.
 */

function engineReturning(
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

const yesNo: DecisionInput = {
  state: "The order was delivered 12 days ago and the item is unopened.",
  question: "Can this order still be returned?",
  options: [
    { id: "yes", description: "Yes" },
    { id: "no", description: "No" },
  ],
};

const approveReject: DecisionInput = {
  state: "The expense is $180 per person with no pre-approval attached.",
  question: "Should this expense be auto-approved?",
  options: [
    { id: "approve", description: "Approve" },
    { id: "reject", description: "Reject" },
  ],
};

const threeWay: DecisionInput = {
  state: "The runbook covers this error and the fix touches no customer data.",
  question: "What should happen to this ticket?",
  options: [
    { id: "escalate", description: "Escalate" },
    { id: "retry", description: "Retry" },
    { id: "complete", description: "Complete" },
  ],
};

describe("answer slots are letters, not words", () => {
  it.each([
    ["Yes/No", yesNo],
    ["Approve/Reject", approveReject],
    ["Escalate/Retry/Complete", threeWay],
  ])("labels %s by position", (_name, input) => {
    const prompt = buildDecisionPrompt(input);
    const labels = ["A", "B", "C"].slice(0, input.options.length);

    for (const [index, label] of labels.entries()) {
      expect(prompt).toContain(`${label}. ${input.options[index]!.description}`);
    }
  });

  it.each([
    ["Yes", yesNo],
    ["Approve", approveReject],
    ["Escalate", threeWay],
  ])("does not let the semantic token %s stand in for its slot", (word, input) => {
    expect(() =>
      resolveOptionProbabilities(input.options, [
        { token: word, logprob: Math.log(0.95) },
        { token: word.toLowerCase(), logprob: Math.log(0.04) },
      ]),
    ).toThrow(DecisionReadoutError);
  });

  it("reads the slot letter even when the option wording starts with another letter", () => {
    const { selected, probabilities } = resolveOptionProbabilities(approveReject.options, [
      { token: "B", logprob: Math.log(0.72) },
      { token: "Approve", logprob: Math.log(0.2) },
      { token: "A", logprob: Math.log(0.05) },
    ]);

    // "Approve" is noise at the answer position; B is the slot, so B wins.
    expect(selected).toBe("reject");
    expect(probabilities.reject).toBeGreaterThan(probabilities.approve!);
  });

  it("maps the third slot to the third option", () => {
    const { selected } = resolveOptionProbabilities(threeWay.options, [
      { token: "C", logprob: Math.log(0.6) },
      { token: "A", logprob: Math.log(0.3) },
      { token: "B", logprob: Math.log(0.1) },
    ]);

    expect(selected).toBe("complete");
  });

  it("ignores a slot letter beyond the declared options", () => {
    const { selected, probabilities } = resolveOptionProbabilities(yesNo.options, [
      { token: "C", logprob: Math.log(0.8) },
      { token: "B", logprob: Math.log(0.15) },
    ]);

    expect(selected).toBe("no");
    expect(Object.keys(probabilities)).toEqual(["yes", "no"]);
  });

  it("keeps identical option wording distinguishable by slot", () => {
    const duplicateWording: DecisionInput = {
      state: "Two teams claim the same ticket.",
      question: "Which team owns it?",
      options: [
        { id: "team-a", description: "Support" },
        { id: "team-b", description: "Support" },
      ],
    };

    const { selected } = resolveOptionProbabilities(duplicateWording.options, [
      { token: "B", logprob: Math.log(0.7) },
      { token: "A", logprob: Math.log(0.3) },
    ]);

    // Identical descriptions; only the slot tells them apart.
    expect(selected).toBe("team-b");
  });

  it("never asks the engine for more than the answer slot", async () => {
    const engine = engineReturning([
      { token: "A", logprob: Math.log(0.8) },
      { token: "B", logprob: Math.log(0.15) },
    ]);
    const create = vi.spyOn(engine.chat.completions, "create");
    const runtime = new WebLLMDecisionRuntime({ createEngine: async () => engine });

    const result = await runtime.decide(approveReject);

    expect(create.mock.calls[0]![0].max_tokens).toBe(1);
    expect(result.trace.readout).toBe("option-logits");
    expect(result.trace.generatedTokens).toBe(0);
    expect(result.selected).toBe("approve");
  });
});
