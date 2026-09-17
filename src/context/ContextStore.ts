export interface ContextChunk {
  id: string;
  text: string;
  source: string;
  metadata: {
    title?: string;
    heading?: string;
  };
}

export interface SearchResult {
  chunk: ContextChunk;
  score: number;
  matchedTerms: string[];
}

interface IndexedChunk {
  chunk: ContextChunk;
  tokens: string[];
  termFrequencies: Map<string, number>;
  normalizedText: string;
  heading: string;
  title: string;
}

const BM25_K1 = 1.5;
const BM25_B = 0.75;

export class ContextStore {
  private readonly indexedChunks: IndexedChunk[] = [];
  private readonly documentFrequency = new Map<string, number>();
  private totalTokenCount = 0;

  add(chunks: ContextChunk[]): void {
    for (const chunk of chunks) {
      const tokens = tokenize(chunk.text);
      const termFrequencies = new Map<string, number>();

      for (const token of tokens) {
        termFrequencies.set(token, (termFrequencies.get(token) ?? 0) + 1);
      }

      for (const token of new Set(tokens)) {
        this.documentFrequency.set(token, (this.documentFrequency.get(token) ?? 0) + 1);
      }

      this.totalTokenCount += tokens.length;
      this.indexedChunks.push({
        chunk,
        tokens,
        termFrequencies,
        normalizedText: normalizeText(chunk.text),
        heading: normalizeText(chunk.metadata.heading ?? ""),
        title: normalizeText(chunk.metadata.title ?? ""),
      });
    }
  }

  clear(): void {
    this.indexedChunks.length = 0;
    this.documentFrequency.clear();
    this.totalTokenCount = 0;
  }

  getChunks(): ContextChunk[] {
    return this.indexedChunks.map(({ chunk }) => chunk);
  }

  search(query: string, limit = 5): SearchResult[] {
    const queryTokens = tokenize(query);

    if (queryTokens.length === 0 || this.indexedChunks.length === 0) {
      return [];
    }

    const normalizedQuery = normalizeText(query);
    const averageLength = this.totalTokenCount / this.indexedChunks.length || 1;
    const uniqueQueryTokens = [...new Set(queryTokens)];

    return this.indexedChunks
      .map((indexedChunk) => {
        let score = 0;
        const matchedTerms: string[] = [];

        for (const token of uniqueQueryTokens) {
          const termFrequency = indexedChunk.termFrequencies.get(token) ?? 0;
          const inHeading = indexedChunk.heading.includes(token);
          const inTitle = indexedChunk.title.includes(token);

          if (!termFrequency && !inHeading && !inTitle) {
            continue;
          }

          matchedTerms.push(token);
          const documentFrequency = this.documentFrequency.get(token) ?? 0;
          const inverseDocumentFrequency = Math.log(
            1 + (this.indexedChunks.length - documentFrequency + 0.5) / (documentFrequency + 0.5),
          );
          const normalizedTermFrequency =
            (termFrequency * (BM25_K1 + 1)) /
            (termFrequency +
              BM25_K1 * (1 - BM25_B + BM25_B * (indexedChunk.tokens.length / averageLength || 1)));

          score += inverseDocumentFrequency * normalizedTermFrequency;

          if (inHeading) {
            score += 1.5;
          }

          if (inTitle) {
            score += 0.5;
          }
        }

        if (normalizedQuery && indexedChunk.normalizedText.includes(normalizedQuery)) {
          score += 2.5;
        }

        if (normalizedQuery && indexedChunk.heading.includes(normalizedQuery)) {
          score += 2;
        }

        return {
          chunk: indexedChunk.chunk,
          score,
          matchedTerms,
        };
      })
      .filter((result) => result.score > 0)
      .sort((left, right) => right.score - left.score)
      .slice(0, limit);
  }
}

function tokenize(text: string): string[] {
  return normalizeText(text)
    .split(/\s+/)
    .filter(Boolean);
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

