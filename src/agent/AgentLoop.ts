import { ContextStore, type ContextChunk } from "../context/ContextStore";
import type { PageSnapshot } from "../tools/getPage";
import { searchContext } from "../tools/searchContext";
import type { Message, ModelRuntime } from "../runtime/ModelRuntime";

export type ContextMode = "entire-page" | "retrieved-context";

export interface AgentTraceEntry {
  type: "task" | "model" | "tool" | "result" | "evidence" | "final";
  content: string;
}

export interface AgentRunInput {
  goal: string;
  page: PageSnapshot;
  mode: ContextMode;
}

export interface AgentRunResult {
  answer: string;
  evidence: ContextChunk[];
  trace: AgentTraceEntry[];
  contextTokens: number;
}

interface ParsedAction {
  action?: "search_context" | "final";
  query?: string;
  answer?: string;
  evidence?: string[];
  note?: string;
}

export interface AgentLoopOptions {
  maxTurns?: number;
  retrievalLimit?: number;
}

const RETRIEVED_CONTEXT_SYSTEM = `You are a browser research agent.
You have access to the current webpage through the search_context tool only.
Do not assume information that you have not retrieved.
Return valid JSON only.

If you need page information, return:
{"action":"search_context","query":"return policy","note":"short visible trace message"}

When you can answer, return:
{"action":"final","answer":"your answer","evidence":["chunk-1"],"note":"short visible trace message"}`;

const ENTIRE_PAGE_SYSTEM = `You are a browser research agent.
You have the current webpage content directly in the prompt.
Do not assume information not present in the page.
Return valid JSON only in this shape:
{"action":"final","answer":"your answer","evidence":["chunk-1"],"note":"short visible trace message"}`;

export class AgentLoop {
  private readonly contextStore = new ContextStore();
  private readonly model: ModelRuntime;
  private readonly options: AgentLoopOptions;

  constructor(model: ModelRuntime, options: AgentLoopOptions = {}) {
    this.model = model;
    this.options = options;
  }

  async run(input: AgentRunInput): Promise<AgentRunResult> {
    this.contextStore.clear();
    this.contextStore.add(input.page.chunks);

    const trace: AgentTraceEntry[] = [{ type: "task", content: input.goal }];

    if (input.mode === "entire-page") {
      return this.runWithEntirePageContext(input, trace);
    }

    return this.runWithRetrievedContext(input, trace);
  }

  private async runWithEntirePageContext(
    input: AgentRunInput,
    trace: AgentTraceEntry[],
  ): Promise<AgentRunResult> {
    const rawResponse = await this.model.generate({
      system: ENTIRE_PAGE_SYSTEM,
      messages: [
        {
          role: "user",
          content: JSON.stringify({
            goal: input.goal,
            page: {
              url: input.page.url,
              title: input.page.title,
              content: input.page.fullText,
              availableEvidence: input.page.chunks.map((chunk) => ({
                id: chunk.id,
                heading: chunk.metadata.heading,
              })),
            },
          }),
        },
      ],
    });
    const parsed = parseAction(rawResponse);
    trace.push({
      type: "model",
      content: parsed.note ?? rawResponse,
    });

    const evidence = resolveEvidence(input.page.chunks, parsed.evidence);
    if (evidence.length > 0) {
      trace.push({
        type: "evidence",
        content: evidence.map(formatEvidence).join("\n"),
      });
    }

    const answer = parsed.answer ?? rawResponse;
    trace.push({ type: "final", content: answer });

    return {
      answer,
      evidence,
      trace,
      contextTokens: estimateTokens(input.page.fullText),
    };
  }

  private async runWithRetrievedContext(
    input: AgentRunInput,
    trace: AgentTraceEntry[],
  ): Promise<AgentRunResult> {
    const messages: Message[] = [
      {
        role: "user",
        content: JSON.stringify({
          goal: input.goal,
          page: {
            url: input.page.url,
            title: input.page.title,
          },
        }),
      },
    ];
    const maxTurns = this.options.maxTurns ?? 4;
    let lastRetrievedContext = "";

    for (let turn = 0; turn < maxTurns; turn += 1) {
      const rawResponse = await this.model.generate({
        system: RETRIEVED_CONTEXT_SYSTEM,
        messages,
      });
      const parsed = parseAction(rawResponse);

      trace.push({
        type: "model",
        content: parsed.note ?? rawResponse,
      });

      if (parsed.action === "search_context" && parsed.query) {
        const toolResult = searchContext(
          this.contextStore,
          parsed.query,
          this.options.retrievalLimit ?? 5,
        );
        lastRetrievedContext = toolResult.results.map(({ chunk }) => chunk.text).join("\n");
        trace.push({
          type: "tool",
          content: `search_context("${parsed.query}")`,
        });
        trace.push({
          type: "result",
          content: toolResult.results
            .map(
              ({ chunk }) =>
                `${chunk.id} (${chunk.metadata.heading ?? chunk.metadata.title ?? "Page"}): ${chunk.text}`,
            )
            .join("\n\n"),
        });

        messages.push({ role: "assistant", content: rawResponse });
        messages.push({
          role: "user",
          content: JSON.stringify({
            tool: "search_context",
            query: parsed.query,
            results: toolResult.results.map(({ chunk, score, matchedTerms }) => ({
              id: chunk.id,
              score,
              matchedTerms,
              source: chunk.source,
              heading: chunk.metadata.heading,
              title: chunk.metadata.title,
              text: chunk.text,
            })),
          }),
        });
        continue;
      }

      const answer = parsed.answer ?? rawResponse;
      const evidence = resolveEvidence(input.page.chunks, parsed.evidence);
      if (evidence.length > 0) {
        trace.push({
          type: "evidence",
          content: evidence.map(formatEvidence).join("\n"),
        });
      }
      trace.push({ type: "final", content: answer });

      return {
        answer,
        evidence,
        trace,
        contextTokens: estimateTokens(lastRetrievedContext),
      };
    }

    trace.push({
      type: "final",
      content: "The agent stopped before producing a final answer.",
    });

    return {
      answer: "The agent stopped before producing a final answer.",
      evidence: [],
      trace,
      contextTokens: estimateTokens(lastRetrievedContext),
    };
  }
}

function parseAction(rawResponse: string): ParsedAction {
  const trimmed = rawResponse.trim();
  const jsonBlockMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = jsonBlockMatch?.[1] ?? trimmed;

  try {
    return JSON.parse(candidate) as ParsedAction;
  } catch {
    return {};
  }
}

function resolveEvidence(chunks: ContextChunk[], evidenceIds: string[] | undefined): ContextChunk[] {
  if (!evidenceIds?.length) {
    return [];
  }

  const chunkById = new Map(chunks.map((chunk) => [chunk.id, chunk]));
  return evidenceIds
    .map((evidenceId) => chunkById.get(evidenceId))
    .filter((chunk): chunk is ContextChunk => Boolean(chunk));
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function formatEvidence(chunk: ContextChunk): string {
  return `${chunk.id} (${chunk.metadata.heading ?? chunk.metadata.title ?? "Page"})`;
}
