import { describe, expect, it } from "vitest";

import { AgentLoop } from "../src/agent/AgentLoop";
import type { PageSnapshot } from "../src/tools/getPage";
import type { ModelRequest, ModelRuntime } from "../src/runtime/ModelRuntime";

class MockModelRuntime implements ModelRuntime {
  private callCount = 0;
  private readonly responses: string[];

  constructor(responses: string[]) {
    this.responses = responses;
  }

  async generate(_request: ModelRequest): Promise<string> {
    const response = this.responses[this.callCount] ?? this.responses[this.responses.length - 1] ?? "";
    this.callCount += 1;
    return response;
  }
}

const page: PageSnapshot = {
  url: "https://example.com/product",
  title: "TrailPack 22L Backpack",
  fullText:
    "TrailPack 22L Backpack Return Policy Items may be returned within 30 days of delivery in original condition.",
  chunks: [
    {
      id: "chunk-1",
      source: "https://example.com/product",
      text: "Items may be returned within 30 days of delivery in original condition.",
      metadata: {
        title: "TrailPack 22L Backpack",
        heading: "Return Policy",
      },
    },
  ],
};

describe("AgentLoop", () => {
  it("retrieves context before answering", async () => {
    const agent = new AgentLoop(
      new MockModelRuntime([
        JSON.stringify({
          action: "search_context",
          query: "return policy",
          note: "I need the return policy.",
        }),
        JSON.stringify({
          action: "final",
          answer: "No. The policy allows returns within 30 days, so 45 days is too late.",
          evidence: ["chunk-1"],
          note: "45 days is outside the allowed window.",
        }),
      ]),
    );

    const result = await agent.run({
      goal: "Find the return policy and tell me whether this product can be returned after 45 days.",
      page,
      mode: "retrieved-context",
    });

    expect(result.answer).toContain("30 days");
    expect(result.evidence.map((chunk) => chunk.id)).toEqual(["chunk-1"]);
    expect(result.trace.map((entry) => entry.type)).toEqual([
      "task",
      "model",
      "tool",
      "result",
      "model",
      "evidence",
      "final",
    ]);
    expect(result.contextTokens).toBeGreaterThan(0);
  });

  it("can answer directly with full page context", async () => {
    const agent = new AgentLoop(
      new MockModelRuntime([
        JSON.stringify({
          action: "final",
          answer: "No. The product cannot be returned after 45 days.",
          evidence: ["chunk-1"],
          note: "The page says returns are limited to 30 days.",
        }),
      ]),
    );

    const result = await agent.run({
      goal: "Can this product be returned after 45 days?",
      page,
      mode: "entire-page",
    });

    expect(result.answer).toContain("cannot be returned");
    expect(result.trace.map((entry) => entry.type)).toEqual(["task", "model", "evidence", "final"]);
    expect(result.contextTokens).toBeGreaterThan(0);
  });
});
