import { describe, expect, it } from "vitest";

import { decideFromContext } from "../src/agent/ContextDecision";
import { ContextStore, type ContextChunk } from "../src/context/ContextStore";
import {
  buildDecisionPrompt,
  type DecisionInput,
  type DecisionResult,
  type DecisionRuntime,
} from "../src/runtime/DecisionRuntime";

/**
 * Stands in for the WebGPU model: records the decision it was handed and reads
 * the answer slot the way the real runtime does.
 */
class RecordingDecisionRuntime implements DecisionRuntime {
  public lastInput?: DecisionInput;
  private readonly pick: (input: DecisionInput) => string;

  constructor(pick: (input: DecisionInput) => string) {
    this.pick = pick;
  }

  async decide(input: DecisionInput): Promise<DecisionResult> {
    this.lastInput = input;
    const selected = this.pick(input);

    return {
      selected,
      probabilities: Object.fromEntries(
        input.options.map((option) => [option.id, option.id === selected ? 0.93 : 0.07]),
      ),
      latencyMs: 61,
      trace: {
        model: "recording",
        execution: "test",
        readout: "option-logits",
        generatedTokens: 0,
        contextTokens: 120,
        options: input.options.length,
        selected,
        optionMass: 0.98,
        probabilityStatus: "test",
        promptVersion: "test",
        promptSha256: "",
        prompt: buildDecisionPrompt(input),
      },
    };
  }
}

/** What `getPage()` produces for a product page, written out directly. */
const pageChunks: ContextChunk[] = [
  {
    id: "chunk-1",
    source: "https://example.com/product",
    text: "Orders ship within 2 business days with free ground shipping.",
    metadata: { title: "TrailPack 22L", heading: "Shipping" },
  },
  {
    id: "chunk-2",
    source: "https://example.com/product",
    text: "Items may be returned within 30 days of delivery in original condition.",
    metadata: { title: "TrailPack 22L", heading: "Return Policy" },
  },
  {
    id: "chunk-3",
    source: "https://example.com/product",
    text: "Final sale accessories cannot be returned.",
    metadata: { title: "TrailPack 22L", heading: "Return Policy" },
  },
];

function pageStore(): ContextStore {
  const store = new ContextStore();
  store.add(pageChunks);
  return store;
}

describe("decideFromContext", () => {
  it("turns retrieved page context into a typed decision", async () => {
    const store = pageStore();
    const runtime = new RecordingDecisionRuntime((input) =>
      input.state.includes("30 days") ? "no" : "yes",
    );

    const { decision, evidence, state } = await decideFromContext(store, runtime, {
      query: "return policy 45 days",
      question: "Can this product be returned after 45 days?",
      options: [
        { id: "yes", description: "The product can be returned." },
        { id: "no", description: "The product cannot be returned." },
      ],
    });

    expect(state).toContain("returned within 30 days");
    expect(evidence.length).toBeGreaterThan(0);
    expect(evidence[0]!.chunk.text).toContain("returned within 30 days");
    expect(decision.selected).toBe("no");
    expect(decision.probabilities.no).toBeGreaterThan(decision.probabilities.yes!);
    expect(decision.trace.generatedTokens).toBe(0);
    expect(decision.trace.readout).toBe("option-logits");
  });

  it("hands the model only the retrieved chunks, not the whole page", async () => {
    const store = pageStore();
    const runtime = new RecordingDecisionRuntime(() => "no");
    await decideFromContext(store, runtime, {
      query: "return policy",
      question: "Can this product be returned after 45 days?",
      options: [
        { id: "yes", description: "The product can be returned." },
        { id: "no", description: "The product cannot be returned." },
      ],
      retrievalLimit: 1,
    });

    expect(runtime.lastInput!.state).toContain("returned within 30 days");
    expect(runtime.lastInput!.state).not.toContain("free ground shipping");
  });
});
