#!/usr/bin/env node
/**
 * Pin the decision models to exact Hugging Face commit revisions.
 *
 * web-llm's prebuilt catalogue resolves weights from the repository's default
 * branch, so a model id alone pins nothing: the same benchmark can silently run
 * on different weights a week later. This rewrites the `revision` fields in
 * src/runtime/decisionModels.ts with the current commit sha of each repo.
 *
 *   npm run pin:model              # pin every model in the registry
 *   npm run pin:model -- --check   # exit non-zero if anything is unpinned
 *
 * Needs network access to huggingface.co.
 */

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const registryPath = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../src/runtime/decisionModels.ts",
);

const source = await readFile(registryPath, "utf8");
const entries = [...source.matchAll(/repo:\s*"([^"]+)",\s*\n\s*revision:\s*([^,]+),/g)].map(
  (match) => ({ repo: match[1], revisionExpression: match[2].trim() }),
);

if (entries.length === 0) {
  console.error(`No model entries found in ${registryPath}.`);
  process.exit(1);
}

const checkOnly = process.argv.includes("--check");

if (checkOnly) {
  const unpinned = entries.filter(
    (entry) => !/^"[0-9a-f]{40}"$/.test(entry.revisionExpression),
  );

  for (const entry of entries) {
    const pinned = /^"[0-9a-f]{40}"$/.test(entry.revisionExpression);
    console.log(`${pinned ? "pinned  " : "UNPINNED"} ${entry.repo} ${entry.revisionExpression}`);
  }

  if (unpinned.length > 0) {
    console.error(`\n${unpinned.length} model(s) unpinned. Run: npm run pin:model`);
    process.exit(1);
  }

  process.exit(0);
}

let updated = source;

for (const entry of entries) {
  const url = `https://huggingface.co/api/models/${entry.repo}`;
  process.stdout.write(`${entry.repo} … `);

  const response = await fetch(url).catch((error) => {
    throw new Error(`${entry.repo}: ${error.message}`);
  });

  if (!response.ok) {
    throw new Error(`${entry.repo}: HTTP ${response.status} from ${url}`);
  }

  const { sha, lastModified } = await response.json();
  if (!/^[0-9a-f]{40}$/.test(sha ?? "")) {
    throw new Error(`${entry.repo}: the API returned no 40-character sha (got ${sha})`);
  }

  const pattern = new RegExp(
    `(repo:\\s*"${entry.repo.replaceAll("/", "\\/")}",\\s*\\n\\s*revision:\\s*)([^,]+)(,)`,
  );
  updated = updated.replace(pattern, `$1"${sha}"$3`);
  console.log(`${sha} (last modified ${lastModified ?? "unknown"})`);
}

await writeFile(registryPath, updated);
console.log(`\nWrote ${registryPath}. Commit it with the benchmark result it produced.`);
