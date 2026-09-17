import {
  DECISION_MODELS,
  GeneratedDecisionRuntime,
  UNPINNED_REVISION,
  WEB_LLM_VERSION,
  WebLLMDecisionRuntime,
  WebLLMRuntime,
  decisionAppConfig,
  describeModelPin,
  findDecisionModel,
  formatDecisionBenchmarkTable,
  runDecisionBenchmark,
  type DecisionBenchmarkPaths,
  type DecisionBenchmarkReport,
  type DecisionFixture,
  type DecisionRunRow,
} from "../src";
import decisionFixturesData from "../benchmark/decisions.json";
import type { AppConfig } from "@mlc-ai/web-llm";

const fixtures = decisionFixturesData as DecisionFixture[];

const modelSelect = requireElement<HTMLSelectElement>("#benchmark-model");
const pathsSelect = requireElement<HTMLSelectElement>("#benchmark-paths");
const limitSelect = requireElement<HTMLSelectElement>("#benchmark-limit");
const runButton = requireElement<HTMLButtonElement>("#benchmark-run");
const status = requireElement<HTMLParagraphElement>("#benchmark-status");
const pinNote = requireElement<HTMLParagraphElement>("#benchmark-pin");
const summaryContainer = requireElement<HTMLDivElement>("#benchmark-summary");
const rowsContainer = requireElement<HTMLDivElement>("#benchmark-rows");
const reportOutput = requireElement<HTMLPreElement>("#benchmark-report");

modelSelect.innerHTML = DECISION_MODELS.map(
  (model) => `<option value="${model.modelId}">${escapeHtml(model.modelId)}</option>`,
).join("");

const showPin = () => {
  const model = findDecisionModel(modelSelect.value);
  pinNote.textContent = model
    ? `${describeModelPin(model)} · web-llm ${WEB_LLM_VERSION}`
    : modelSelect.value;
};

modelSelect.addEventListener("change", showPin);
runButton.addEventListener("click", () => {
  void run();
});
showPin();

async function run(): Promise<void> {
  const modelId = modelSelect.value;
  const selection = pathsSelect.value;
  const limit = Number(limitSelect.value);
  const selectedFixtures = fixtures.slice(0, limit);

  runButton.disabled = true;
  summaryContainer.innerHTML = "";
  rowsContainer.innerHTML = "";
  reportOutput.textContent = "";
  status.textContent = "Loading the model…";

  try {
    // One engine, both paths: the comparison is only meaningful if the
    // generated answer and the logit readout come from the same weights.
    const engine = await createSharedEngine(modelId, (text) => {
      status.textContent = text;
    });

    const paths: DecisionBenchmarkPaths = {};
    if (selection !== "direct") {
      paths.generated = new GeneratedDecisionRuntime(
        new WebLLMRuntime({ model: modelId, createEngine: async () => engine }),
        { model: modelId },
      );
    }
    if (selection !== "generated") {
      paths.direct = new WebLLMDecisionRuntime({ model: modelId, createEngine: async () => engine });
    }

    const rows: DecisionRunRow[] = [];
    const report = await runDecisionBenchmark({
      model: modelId,
      fixtures: selectedFixtures,
      paths,
      onProgress: ({ completed, total, row }) => {
        rows.push(row);
        status.textContent = `${completed} / ${total} runs`;
        renderRows(rows);
      },
    });

    renderSummary(report);
    renderRows(report.rows);
    reportOutput.textContent = [
      formatDecisionBenchmarkTable(report),
      "",
      `web-llm: ${WEB_LLM_VERSION}`,
      describeModelPin(
        findDecisionModel(modelId) ?? {
          modelId,
          repo: "(not in the pinned registry)",
          revision: UNPINNED_REVISION,
          modelLib: "(web-llm prebuilt)",
          vramRequiredMB: 0,
          contextWindowSize: 0,
        },
      ),
      "",
      JSON.stringify(report.rows, null, 2),
    ].join("\n");
    status.textContent = "Done.";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Unknown error";
  } finally {
    runButton.disabled = false;
  }
}

/**
 * The engine both paths run on.
 *
 * web-llm's `MLCEngine` satisfies the narrow engine interfaces `WebLLMRuntime`
 * and `WebLLMDecisionRuntime` each declare; this type names the overlap so the
 * same instance can be handed to both without loading the weights twice.
 */
interface SharedEngine {
  reload(model: string): Promise<void>;
  getGPUVendor?(): Promise<string>;
  chat: {
    completions: {
      create(request: {
        messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
        temperature?: number;
        max_tokens?: number;
        logprobs?: boolean;
        top_logprobs?: number;
      }): Promise<{
        choices?: Array<{
          message?: { content?: string | Array<{ text?: string }> };
          logprobs?: {
            content?: Array<{
              token?: string;
              logprob?: number;
              top_logprobs?: Array<{ token?: string; logprob?: number }>;
            }> | null;
          } | null;
        }>;
        usage?: { prompt_tokens?: number };
      }>;
    };
  };
}

async function createSharedEngine(
  modelId: string,
  onProgress: (text: string) => void,
): Promise<SharedEngine> {
  const webllm = await import("@mlc-ai/web-llm");
  const pinned = findDecisionModel(modelId);
  const engine = new webllm.MLCEngine({
    initProgressCallback: (progress) => onProgress(progress.text ?? "Loading model…"),
    appConfig: pinned ? (decisionAppConfig([pinned]) as AppConfig) : undefined,
  });
  await engine.reload(modelId);

  // The engine's `create` is overloaded for streaming; this narrows it to the
  // non-streaming shape both runtimes ask for.
  return engine as unknown as SharedEngine;
}

function renderSummary(report: DecisionBenchmarkReport): void {
  const labels: Record<string, string> = { generated: "Generated", direct: "Direct logits" };
  const body = report.summaries
    .map(
      (summary) => `
        <tr>
          <td>${labels[summary.path] ?? summary.path}</td>
          <td>${(summary.accuracy * 100).toFixed(0)}% (${summary.correct}/${summary.runs})</td>
          <td>${Math.round(summary.medianLatencyMs)} ms</td>
          <td>${summary.totalGeneratedTokens}</td>
          <td>${Math.round(summary.medianContextTokens)}</td>
          <td>${summary.errors}</td>
        </tr>
      `,
    )
    .join("");

  summaryContainer.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Path</th>
          <th>Accuracy</th>
          <th>Median latency</th>
          <th>Output tokens</th>
          <th>Median context tokens</th>
          <th>Errors</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
    <p class="muted">
      ${
        report.comparedFixtures > 0
          ? `The two paths agreed on ${(report.agreement * 100).toFixed(0)}% of ${
              report.comparedFixtures
            } compared fixtures. This is a systems comparison, not a claim that the two readouts are semantically equivalent.`
          : "Only one path ran, so there is no agreement figure."
      }
    </p>
  `;
}

function renderRows(rows: DecisionRunRow[]): void {
  const body = rows
    .map(
      (row) => `
        <tr class="${row.correct ? "" : "wrong"}">
          <td>${escapeHtml(row.fixtureId)}</td>
          <td>${escapeHtml(row.path)}</td>
          <td>${escapeHtml(row.selected ?? "—")}</td>
          <td>${escapeHtml(row.expected)}</td>
          <td>${row.correct ? "✓" : "✗"}</td>
          <td>${Math.round(row.latencyMs)} ms</td>
          <td>${row.generatedTokens}</td>
          <td>${row.optionMass === undefined ? "—" : row.optionMass.toFixed(3)}</td>
          <td class="muted">${escapeHtml(row.error ?? "")}</td>
        </tr>
      `,
    )
    .join("");

  rowsContainer.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Fixture</th>
          <th>Path</th>
          <th>Selected</th>
          <th>Expected</th>
          <th>Correct</th>
          <th>Latency</th>
          <th>Output tokens</th>
          <th>Option mass</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Benchmark page is missing ${selector}.`);
  }

  return element;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
