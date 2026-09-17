import type { ContextStore, SearchResult } from "../context/ContextStore";
import { searchContext } from "../tools/searchContext";
import type {
  DecisionOption,
  DecisionResult,
  DecisionRuntime,
} from "../runtime/DecisionRuntime";

export interface ContextDecisionInput {
  /** Retrieval query for the page context the decision should be taken over. */
  query: string;
  question: string;
  options: DecisionOption[];
  retrievalLimit?: number;
}

export interface ContextDecisionResult {
  decision: DecisionResult;
  /** The chunks that became the decision's state, in retrieval order. */
  evidence: SearchResult[];
  state: string;
}

/**
 * Context → Decision.
 *
 * The same retrieval the agent loop uses, feeding a typed decision instead of a
 * generated answer:
 *
 *   page → ContextStore → search_context → DecisionRuntime → typed decision
 */
export async function decideFromContext(
  contextStore: ContextStore,
  decisionRuntime: DecisionRuntime,
  input: ContextDecisionInput,
): Promise<ContextDecisionResult> {
  const retrieved = searchContext(contextStore, input.query, input.retrievalLimit ?? 5);
  const state = retrieved.results.map((result) => result.chunk.text).join("\n");

  const decision = await decisionRuntime.decide({
    state,
    question: input.question,
    options: input.options,
  });

  return { decision, evidence: retrieved.results, state };
}
