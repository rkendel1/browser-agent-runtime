import type { ContextStore, SearchResult } from "../context/ContextStore";

export interface SearchContextToolResult {
  query: string;
  results: SearchResult[];
}

export function searchContext(
  contextStore: ContextStore,
  query: string,
  limit = 5,
): SearchContextToolResult {
  return {
    query,
    results: contextStore.search(query, limit),
  };
}

