import { describe, expect, it } from "vitest";

import { ContextStore, type ContextChunk } from "../src/context/ContextStore";

describe("ContextStore", () => {
  it("prioritizes the return policy chunk for lexical queries", () => {
    const chunks: ContextChunk[] = [
      {
        id: "chunk-1",
        source: "https://example.com/product",
        text: "Items may be returned within 30 days of delivery in original condition.",
        metadata: {
          title: "Product page",
          heading: "Return Policy",
        },
      },
      {
        id: "chunk-2",
        source: "https://example.com/product",
        text: "Orders ship within 2 business days with free ground shipping.",
        metadata: {
          title: "Product page",
          heading: "Shipping",
        },
      },
    ];

    const store = new ContextStore();
    store.add(chunks);

    const results = store.search("return policy 45 days");

    expect(results[0]?.chunk.id).toBe("chunk-1");
    expect(results[0]?.matchedTerms).toContain("return");
    expect(results[0]?.matchedTerms).toContain("days");
  });
});

