import { chunkDocument, normalizeWhitespace } from "../context/ContextResolver";
import type { ContextChunk } from "../context/ContextStore";

export interface PageSnapshot {
  url: string;
  title: string;
  fullText: string;
  chunks: ContextChunk[];
}

export function getPage(document: Document): PageSnapshot {
  const url = document.location?.href ?? "about:blank";
  const title = normalizeWhitespace(document.title) || url;
  const fullText = normalizeWhitespace(document.body?.innerText ?? "");
  const chunks = chunkDocument(document, url);

  return {
    url,
    title,
    fullText,
    chunks,
  };
}

