import type { ContextChunk } from "./ContextStore";

interface TextSection {
  heading?: string;
  text: string;
}

export interface ChunkTextOptions {
  source: string;
  title?: string;
  heading?: string;
  maxCharacters?: number;
}

export function chunkDocument(document: Document, source = document.location?.href ?? "about:blank"): ContextChunk[] {
  const title = normalizeWhitespace(document.title) || "Untitled page";
  const blocks = Array.from(document.body?.querySelectorAll("h1,h2,h3,h4,h5,h6,p,li") ?? []);

  if (blocks.length === 0) {
    return chunkText(document.body?.innerText ?? "", { source, title });
  }

  const sections: TextSection[] = [];
  let currentHeading = title;
  let buffer: string[] = [];

  const flush = () => {
    const text = normalizeWhitespace(buffer.join("\n"));
    if (!text) {
      buffer = [];
      return;
    }

    sections.push({ heading: currentHeading, text });
    buffer = [];
  };

  for (const block of blocks) {
    const text = normalizeWhitespace(block.textContent ?? "");
    if (!text) {
      continue;
    }

    if (/^H[1-6]$/.test(block.tagName)) {
      flush();
      currentHeading = text;
      continue;
    }

    buffer.push(text);
  }

  flush();

  if (sections.length === 0) {
    return chunkText(document.body?.innerText ?? "", { source, title });
  }

  return sections
    .flatMap((section) =>
      chunkText(section.text, {
        source,
        title,
        heading: section.heading,
      }),
    )
    .map((chunk, index) => ({
      ...chunk,
      id: `chunk-${index + 1}`,
    }));
}

export function chunkText(text: string, options: ChunkTextOptions): ContextChunk[] {
  const normalized = normalizeWhitespace(text);
  if (!normalized) {
    return [];
  }

  const maxCharacters = options.maxCharacters ?? 900;
  const units = normalized.split(/(?<=[.!?])\s+|\n+/).map(normalizeWhitespace).filter(Boolean);
  const chunks: ContextChunk[] = [];
  let currentChunk: string[] = [];
  let currentLength = 0;
  let index = 1;

  const flush = () => {
    const chunkTextValue = normalizeWhitespace(currentChunk.join(" "));
    if (!chunkTextValue) {
      currentChunk = [];
      currentLength = 0;
      return;
    }

    chunks.push({
      id: `chunk-${index++}`,
      text: chunkTextValue,
      source: options.source,
      metadata: {
        title: options.title,
        heading: options.heading,
      },
    });
    currentChunk = [];
    currentLength = 0;
  };

  for (const unit of units) {
    if (currentLength > 0 && currentLength + unit.length + 1 > maxCharacters) {
      flush();
    }

    currentChunk.push(unit);
    currentLength += unit.length + 1;
  }

  flush();
  return chunks;
}

export function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}
