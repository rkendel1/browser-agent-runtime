import type { AppConfig } from "@mlc-ai/web-llm";

import {
  DECISION_MODELS,
  DECISION_PROBABILITY_STATUS,
  DECISION_PROMPT_VERSION,
  DECISION_READOUT_TEMPERATURE,
  GeneratedDecisionRuntime,
  LOW_OPTION_MASS_THRESHOLD,
  MAX_TOP_LOGPROBS,
  WEB_LLM_VERSION,
  WebLLMDecisionRuntime,
  WebLLMRuntime,
  buildBenchmarkArtifact,
  decisionAppConfig,
  describeModelPin,
  findDecisionModel,
  formatDecisionBenchmarkReport,
  formatDecisionBenchmarkTable,
  isPinnedRevision,
  runDecisionBenchmark,
  runOptionMassProbes,
  type DecisionBenchmarkArtifact,
  type DecisionCondition,
  type DecisionContextSource,
  type DecisionFixture,
  type DecisionPathName,
  type DecisionRunRow,
  type OptionMassProbe,
  type ProbeRunRow,
} from "../src";
import decisionFixturesData from "../benchmark/decisions.json";
import probeData from "../benchmark/option-mass-probes.json";
import { captureEnvironment, createStateResolver } from "./benchmarkContext";
import { createStubEngine } from "./stubEngine";

const fixtures = decisionFixturesData as DecisionFixture[];
const probes = probeData as OptionMassProbe[];

const RETRIEVAL_LIMIT = 5;

/** The experiment: two readouts across three context sources. */
const CONDITION_CHOICES: Array<{
  path: DecisionPathName;
  context: DecisionContextSource;
  label: string;
  checked: boolean;
}> = [
  { path: "generated", context: "state", label: "Generated · state only (control)", checked: true },
  { path: "direct", context: "state", label: "Direct · state only (control)", checked: true },
  { path: "generated", context: "full-page", label: "Generated · full page (A)", checked: true },
  { path: "generated", context: "retrieved", label: "Generated · retrieved (B)", checked: true },
  { path: "direct", context: "full-page", label: "Direct · full page (C)", checked: true },
  { path: "direct", context: "retrieved", label: "Direct · retrieved (D)", checked: true },
];

const modelSelect = requireElement<HTMLSelectElement>("#benchmark-model");
const conditionList = requireElement<HTMLDivElement>("#benchmark-conditions");
const limitSelect = requireElement<HTMLSelectElement>("#benchmark-limit");
const probesToggle = requireElement<HTMLInputElement>("#benchmark-probes");
const runButton = requireElement<HTMLButtonElement>("#benchmark-run");
const exportButton = requireElement<HTMLButtonElement>("#benchmark-export");
const copyButton = requireElement<HTMLButtonElement>("#benchmark-copy");
const status = requireElement<HTMLParagraphElement>("#benchmark-status");
const pinNote = requireElement<HTMLParagraphElement>("#benchmark-pin");
const engineNote = requireElement<HTMLParagraphElement>("#benchmark-engine");
const reportOutput = requireElement<HTMLPreElement>("#benchmark-report");
const summaryContainer = requireElement<HTMLDivElement>("#benchmark-summary");
const comparisonContainer = requireElement<HTMLDivElement>("#benchmark-comparison");
const rowsContainer = requireElement<HTMLDivElement>("#benchmark-rows");
const probesContainer = requireElement<HTMLDivElement>("#benchmark-probes-results");

/**
 * `?engine=stub` swaps the model for a deterministic fake so the apparatus can
 * be exercised without downloading weights. Every artifact from such a run is
 * stamped `"engine": "stub"`.
 */
const useStubEngine = new URLSearchParams(location.search).get("engine") === "stub";

let artifact: DecisionBenchmarkArtifact | undefined;

initialize();

function initialize(): void {
  modelSelect.innerHTML = DECISION_MODELS.map(
    (model) => `<option value="${model.modelId}">${escapeHtml(model.modelId)}</option>`,
  ).join("");

  conditionList.innerHTML = CONDITION_CHOICES.map(
    (choice) => `
      <label class="checkbox-row">
        <input type="checkbox" data-path="${choice.path}" data-context="${choice.context}" ${
          choice.checked ? "checked" : ""
        } />
        ${escapeHtml(choice.label)}
      </label>
    `,
  ).join("");

  if (useStubEngine) {
    engineNote.textContent =
      "STUB ENGINE: no model is loaded and every number below is fabricated by demo/stubEngine.ts. " +
      "This mode exists to test the benchmark itself.";
    engineNote.classList.add("warning");
  }

  modelSelect.addEventListener("change", showPin);
  runButton.addEventListener("click", () => {
    void run();
  });
  exportButton.addEventListener("click", exportJson);
  copyButton.addEventListener("click", () => {
    void copyJson();
  });

  showPin();
  setExportEnabled(false);
}

function showPin(): void {
  const model = findDecisionModel(modelSelect.value);
  pinNote.textContent = model
    ? `${describeModelPin(model)} · web-llm ${WEB_LLM_VERSION} · prompt ${DECISION_PROMPT_VERSION}`
    : modelSelect.value;
  pinNote.classList.toggle("warning", Boolean(model && !isPinnedRevision(model.revision)));
}

function selectedConditions(): Array<{ path: DecisionPathName; context: DecisionContextSource }> {
  return [...conditionList.querySelectorAll<HTMLInputElement>("input[type=checkbox]")]
    .filter((input) => input.checked)
    .map((input) => ({
      path: input.dataset.path as DecisionPathName,
      context: input.dataset.context as DecisionContextSource,
    }));
}

async function run(): Promise<void> {
  const modelId = modelSelect.value;
  const chosen = selectedConditions();
  const selectedFixtures = fixtures.slice(0, Number(limitSelect.value));

  if (chosen.length === 0) {
    status.textContent = "Select at least one condition.";
    return;
  }

  runButton.disabled = true;
  setExportEnabled(false);
  summaryContainer.innerHTML = "";
  comparisonContainer.innerHTML = "";
  rowsContainer.innerHTML = "";
  probesContainer.innerHTML = "";
  reportOutput.textContent = "";
  status.textContent = useStubEngine ? "Starting stub run…" : "Loading the model…";

  try {
    // One engine for every condition: the comparison is only meaningful if the
    // generated answer and the logit readout come from the same weights in the
    // same browser.
    const engine = await createSharedEngine(modelId, (text) => {
      status.textContent = text;
    });

    const directRuntime = new WebLLMDecisionRuntime({
      model: modelId,
      createEngine: async () => engine,
    });
    const generatedRuntime = new GeneratedDecisionRuntime(
      new WebLLMRuntime({ model: modelId, createEngine: async () => engine }),
      { model: modelId, execution: useStubEngine ? "stub" : "webgpu" },
    );

    const conditions: DecisionCondition[] = chosen.map((condition) => ({
      ...condition,
      runtime: condition.path === "direct" ? directRuntime : generatedRuntime,
    }));

    const liveRows: DecisionRunRow[] = [];
    const { result, rows } = await runDecisionBenchmark({
      fixtures: selectedFixtures,
      conditions,
      resolveState: createStateResolver(RETRIEVAL_LIMIT),
      onProgress: ({ completed, total, row }) => {
        liveRows.push(row);
        status.textContent = `${completed} / ${total} runs`;
        renderRows(liveRows);
      },
    });

    let probeRows: ProbeRunRow[] = [];
    if (probesToggle.checked) {
      status.textContent = "Running option mass probes…";
      probeRows = await runOptionMassProbes({
        probes,
        runtime: directRuntime,
        onProgress: ({ completed, total }) => {
          status.textContent = `probe ${completed} / ${total}`;
        },
      });
    }

    const model = findDecisionModel(modelId);
    artifact = buildBenchmarkArtifact({
      environment: await captureEnvironment(useStubEngine ? "stub" : "webgpu"),
      model: {
        modelId,
        repo: model?.repo,
        revision: model?.revision,
        pinned: model ? isPinnedRevision(model.revision) : false,
        modelLib: model?.modelLib,
        webllmVersion: WEB_LLM_VERSION,
        vramRequiredMB: model?.vramRequiredMB,
        contextWindowSize: model?.contextWindowSize,
      },
      runtime: {
        promptVersion: DECISION_PROMPT_VERSION,
        readoutTemperature: DECISION_READOUT_TEMPERATURE,
        generationTemperature: 0,
        maxTokens: 1,
        topLogprobs: MAX_TOP_LOGPROBS,
        retrievalLimit: RETRIEVAL_LIMIT,
        lowOptionMassThreshold: LOW_OPTION_MASS_THRESHOLD,
        probabilityStatus: DECISION_PROBABILITY_STATUS,
      },
      benchmark: result,
      results: rows,
      probes: probeRows,
    });

    renderSummary(artifact);
    renderComparison(artifact);
    renderRows(rows);
    renderProbes(probeRows);
    reportOutput.textContent = `${formatDecisionBenchmarkReport(artifact)}\n\n${formatDecisionBenchmarkTable(
      artifact,
    )}`;
    setExportEnabled(true);
    status.textContent = `Done in ${(result.durationMs / 1000).toFixed(1)}s.`;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Unknown error";
  } finally {
    runButton.disabled = false;
  }
}

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
  if (useStubEngine) {
    return createStubEngine() as unknown as SharedEngine;
  }

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

function renderSummary(current: DecisionBenchmarkArtifact): void {
  const body = current.benchmark.summaries
    .map(
      (summary) => `
        <tr>
          <td>${escapeHtml(summary.conditionId)}</td>
          <td>${(summary.accuracy * 100).toFixed(0)}% (${summary.correct}/${summary.runs})</td>
          <td>${Math.round(summary.latencyMs.median)} ms</td>
          <td>${summary.generatedTokens.median}</td>
          <td>${Math.round(summary.contextTokens.median)}</td>
          <td>${summary.optionMass ? summary.optionMass.median.toFixed(2) : "—"}</td>
          <td>${summary.lowOptionMassCount}</td>
          <td>${summary.unreadableSlots}</td>
          <td>${summary.errors}</td>
          <td>${
            summary.retrievalHitRate === undefined
              ? "—"
              : `${(summary.retrievalHitRate * 100).toFixed(0)}%`
          }</td>
        </tr>
      `,
    )
    .join("");

  const agreements = current.benchmark.agreements
    .map(
      (entry) =>
        `<li>${escapeHtml(entry.context)}: ${(entry.agreement * 100).toFixed(0)}% (${entry.agreed}/${
          entry.comparedFixtures
        })</li>`,
    )
    .join("");

  summaryContainer.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Condition</th>
          <th>Accuracy</th>
          <th>Median latency</th>
          <th>Median output</th>
          <th>Median context</th>
          <th>Option mass</th>
          <th>Low mass</th>
          <th>Unreadable</th>
          <th>Errors</th>
          <th>Retrieval hit</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
    <p class="muted">Agreement between the paths, per context source:</p>
    <ul class="muted">${agreements || "<li>only one path ran</li>"}</ul>
    <p class="muted">
      Option mass is the probability the model left on the option labels — not confidence. The
      direct path reports P(option label | prompt); the generated path reports a parsed choice.
      Agreement is a systems comparison, not semantic equivalence.
    </p>
  `;
}

/** Per fixture: what each readout said, side by side. */
function renderComparison(current: DecisionBenchmarkArtifact): void {
  const contexts = [...new Set(current.results.map((row) => row.context))];

  comparisonContainer.innerHTML = contexts
    .map((context) => {
      const rows = fixtures
        .map((fixture) => {
          const direct = current.results.find(
            (row) =>
              row.fixtureId === fixture.id && row.context === context && row.path === "direct",
          );
          const generated = current.results.find(
            (row) =>
              row.fixtureId === fixture.id && row.context === context && row.path === "generated",
          );

          if (!direct && !generated) {
            return "";
          }

          const agree =
            direct?.selected && generated?.selected
              ? direct.selected === generated.selected
                ? "yes"
                : "no"
              : "—";

          return `
            <tr>
              <td>${escapeHtml(fixture.id)}</td>
              <td>${escapeHtml(fixture.expected)}</td>
              <td>${escapeHtml(direct?.selected ?? "—")}</td>
              <td>${direct?.optionMass === undefined ? "—" : direct.optionMass.toFixed(2)}${
                direct?.lowOptionMass ? " ⚠" : ""
              }</td>
              <td>${direct ? `${Math.round(direct.latencyMs)} ms` : "—"}</td>
              <td>${escapeHtml(generated?.selected ?? "—")}</td>
              <td>${generated ? `${Math.round(generated.latencyMs)} ms` : "—"}</td>
              <td>${generated ? generated.generatedTokens : "—"}</td>
              <td>${agree}</td>
            </tr>
          `;
        })
        .join("");

      return `
        <h3>${escapeHtml(context)}</h3>
        <table>
          <thead>
            <tr>
              <th rowspan="2">Fixture</th>
              <th rowspan="2">Expected</th>
              <th colspan="3">Direct logits</th>
              <th colspan="3">Generated</th>
              <th rowspan="2">Agreement</th>
            </tr>
            <tr>
              <th>selected</th>
              <th>option mass</th>
              <th>latency</th>
              <th>selected</th>
              <th>latency</th>
              <th>output tokens</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    })
    .join("");
}

function renderRows(rows: DecisionRunRow[]): void {
  const body = rows
    .map(
      (row) => `
        <tr class="${row.correct ? "" : "wrong"}">
          <td>${escapeHtml(row.fixtureId)}</td>
          <td>${escapeHtml(row.conditionId)}</td>
          <td>${escapeHtml(row.selected ?? "—")}</td>
          <td>${escapeHtml(row.expected)}</td>
          <td>${row.correct ? "✓" : "✗"}</td>
          <td>${Math.round(row.latencyMs)} ms</td>
          <td>${row.generatedTokens}</td>
          <td>${row.contextTokens}</td>
          <td>${row.optionMass === undefined ? "—" : row.optionMass.toFixed(3)}${
            row.lowOptionMass ? " ⚠" : ""
          }</td>
          <td>${
            row.retrievedStateHit === undefined ? "—" : row.retrievedStateHit ? "hit" : "miss"
          }</td>
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
          <th>Condition</th>
          <th>Selected</th>
          <th>Expected</th>
          <th>Correct</th>
          <th>Latency</th>
          <th>Output</th>
          <th>Context</th>
          <th>Option mass</th>
          <th>Retrieval</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderProbes(rows: ProbeRunRow[]): void {
  if (rows.length === 0) {
    probesContainer.innerHTML = `<p class="muted">Probes were not run.</p>`;
    return;
  }

  const body = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.probeId)}</td>
          <td>${escapeHtml(row.selected ?? "—")}</td>
          <td>${
            row.probabilities
              ? escapeHtml(
                  Object.entries(row.probabilities)
                    .map(([id, probability]) => `${id} ${(probability * 100).toFixed(0)}%`)
                    .join(", "),
                )
              : "—"
          }</td>
          <td>${row.optionMass === undefined ? "—" : row.optionMass.toFixed(3)}</td>
          <td>${row.lowOptionMass ? "yes" : "no"}</td>
          <td class="muted">${escapeHtml(row.error ?? row.why)}</td>
        </tr>
      `,
    )
    .join("");

  probesContainer.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Probe</th>
          <th>Selected</th>
          <th>Probabilities</th>
          <th>Option mass</th>
          <th>Low mass</th>
          <th>Why this probe</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
    <p class="muted">
      Probes have no correct answer. A high probability next to a low option mass is the failure
      mode being measured: the normalized numbers look decisive while the model placed most of its
      next-token mass somewhere other than the option labels.
    </p>
  `;
}

function setExportEnabled(enabled: boolean): void {
  exportButton.disabled = !enabled;
  copyButton.disabled = !enabled;
}

function artifactFilename(current: DecisionBenchmarkArtifact): string {
  const stamp = current.environment.timestamp.replace(/[:.]/g, "-");
  const engine = current.environment.engine === "stub" ? "-stub" : "";
  return `decision-benchmark-${current.model.modelId}${engine}-${stamp}.json`;
}

function exportJson(): void {
  if (!artifact) {
    return;
  }

  const blob = new Blob([JSON.stringify(artifact, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = artifactFilename(artifact);
  // Some browsers only follow a link that is in the document, and revoking the
  // URL in the same tick can cancel the download before it starts.
  document.body.append(link);
  link.click();
  setTimeout(() => {
    link.remove();
    URL.revokeObjectURL(url);
  }, 0);
}

async function copyJson(): Promise<void> {
  if (!artifact) {
    return;
  }

  try {
    await navigator.clipboard.writeText(JSON.stringify(artifact, null, 2));
    status.textContent = "Report JSON copied to the clipboard.";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not copy the report.";
  }
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
