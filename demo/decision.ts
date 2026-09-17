import DOMPurify from "dompurify";

import {
  ContextStore,
  DECISION_MODELS,
  WebLLMDecisionRuntime,
  decideFromContext,
  describeModelPin,
  findDecisionModel,
  getPage,
  type DecisionOption,
  type DecisionResult,
  type DecisionRuntime,
  type SearchResult,
} from "../src";
import decisionFixturesData from "../benchmark/decisions.json";
import type { DecisionFixture } from "../src/benchmark/DecisionBenchmark";

const decisionFixtures = decisionFixturesData as DecisionFixture[];

const runtimeByModel = new Map<string, DecisionRuntime>();

export function initializeDecisionPanel(getPageHtml: () => string): void {
  const fixtureSelect = requireElement<HTMLSelectElement>("#decision-fixture");
  const modelSelect = requireElement<HTMLSelectElement>("#decision-model");
  const modelPin = requireElement<HTMLParagraphElement>("#decision-model-pin");
  const stateInput = requireElement<HTMLTextAreaElement>("#decision-state");
  const questionInput = requireElement<HTMLInputElement>("#decision-question");
  const optionsInput = requireElement<HTMLTextAreaElement>("#decision-options");
  const useContext = requireElement<HTMLInputElement>("#decision-use-context");
  const queryInput = requireElement<HTMLInputElement>("#decision-query");
  const decideButton = requireElement<HTMLButtonElement>("#decide-button");
  const status = requireElement<HTMLParagraphElement>("#decision-status");
  const results = requireElement<HTMLDivElement>("#decision-results");
  const traceOutput = requireElement<HTMLPreElement>("#decision-trace");

  fixtureSelect.innerHTML = decisionFixtures
    .map((fixture) => `<option value="${fixture.id}">${escapeHtml(fixture.name)}</option>`)
    .join("");

  modelSelect.innerHTML = DECISION_MODELS.map(
    (model) => `<option value="${model.modelId}">${escapeHtml(model.modelId)}</option>`,
  ).join("");

  const showModelPin = () => {
    const model = findDecisionModel(modelSelect.value);
    modelPin.textContent = model ? describeModelPin(model) : modelSelect.value;
  };

  const loadFixture = (fixtureId: string) => {
    const fixture =
      decisionFixtures.find((candidate) => candidate.id === fixtureId) ?? decisionFixtures[0];
    if (!fixture) {
      return;
    }

    stateInput.value = fixture.state;
    questionInput.value = fixture.question;
    optionsInput.value = fixture.options
      .map((option) => `${option.id} | ${option.description}`)
      .join("\n");
    status.textContent = `Expected: ${fixture.expected}`;
    results.innerHTML = "";
    traceOutput.textContent = "";
  };

  const syncContextControls = () => {
    queryInput.disabled = !useContext.checked;
    stateInput.disabled = useContext.checked;
    stateInput.title = useContext.checked
      ? "State comes from the retrieved page context while this box is checked."
      : "";
  };

  fixtureSelect.addEventListener("change", () => loadFixture(fixtureSelect.value));
  modelSelect.addEventListener("change", showModelPin);
  useContext.addEventListener("change", syncContextControls);
  decideButton.addEventListener("click", () => {
    void decide();
  });

  loadFixture(decisionFixtures[0]?.id ?? "");
  showModelPin();
  syncContextControls();

  async function decide(): Promise<void> {
    let options: DecisionOption[];
    try {
      options = parseOptions(optionsInput.value);
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : String(error);
      return;
    }

    decideButton.disabled = true;
    status.textContent = "Loading model and reading option logits…";
    results.innerHTML = "";
    traceOutput.textContent = "";

    try {
      const runtime = getRuntime(modelSelect.value, (text) => {
        status.textContent = text;
      });

      let decision: DecisionResult;
      let evidence: SearchResult[] = [];

      if (useContext.checked) {
        // page → ContextStore → search_context → DecisionRuntime → typed decision
        const store = new ContextStore();
        store.add(parsePage(getPageHtml()).chunks);
        const fromContext = await decideFromContext(store, runtime, {
          query: queryInput.value.trim() || questionInput.value.trim(),
          question: questionInput.value.trim(),
          options,
        });
        decision = fromContext.decision;
        evidence = fromContext.evidence;
        stateInput.value = fromContext.state;
      } else {
        decision = await runtime.decide({
          state: stateInput.value,
          question: questionInput.value.trim(),
          options,
        });
      }

      renderDecision(results, options, decision);
      traceOutput.textContent = formatTrace(decision, evidence);
      status.textContent = `Selected: ${decision.selected}`;
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Unknown error";
    } finally {
      decideButton.disabled = false;
    }
  }
}

function getRuntime(modelId: string, onProgress: (text: string) => void): DecisionRuntime {
  const existing = runtimeByModel.get(modelId);
  if (existing) {
    return existing;
  }

  const runtime = new WebLLMDecisionRuntime({
    model: modelId,
    initProgressCallback: (progress) => onProgress(progress.text ?? "Loading model…"),
  });
  runtimeByModel.set(modelId, runtime);
  return runtime;
}

/** One option per line: `id | description`. */
export function parseOptions(value: string): DecisionOption[] {
  const options = value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const separator = line.indexOf("|");
      if (separator < 0) {
        throw new Error(`Each option needs an id and a description: "${line}"`);
      }

      return {
        id: line.slice(0, separator).trim(),
        description: line.slice(separator + 1).trim(),
      };
    });

  if (options.length < 2) {
    throw new Error("A decision needs at least two options.");
  }

  return options;
}

function parsePage(html: string) {
  const sanitized = DOMPurify.sanitize(html, { WHOLE_DOCUMENT: true });
  return getPage(new DOMParser().parseFromString(sanitized, "text/html"));
}

function renderDecision(
  container: HTMLDivElement,
  options: DecisionOption[],
  decision: DecisionResult,
): void {
  const bars = options
    .map((option) => {
      const probability = decision.probabilities[option.id] ?? 0;
      const percent = (probability * 100).toFixed(probability >= 0.995 || probability === 0 ? 0 : 1);
      const selected = option.id === decision.selected;

      return `
        <div class="probability${selected ? " probability-selected" : ""}">
          <div class="probability-label">
            <span>${escapeHtml(option.id)}</span>
            <span>${percent}%</span>
          </div>
          <div class="probability-track"><div class="probability-fill" style="width: ${(
            probability * 100
          ).toFixed(2)}%"></div></div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = `
    ${bars}
    <div class="decision-summary">
      <div><strong>Selected</strong> ${escapeHtml(decision.selected)}</div>
      <div><strong>Latency</strong> ${Math.round(decision.latencyMs)}ms</div>
    </div>
  `;
}

function formatTrace(decision: DecisionResult, evidence: SearchResult[]): string {
  const { trace } = decision;
  const lines = [
    `model: ${trace.model}`,
    `execution: ${trace.execution}`,
    `readout: ${trace.readout}`,
    `generated_tokens: ${trace.generatedTokens}`,
    `context_tokens: ${trace.contextTokens}`,
    `options: ${trace.options}`,
    `selected: ${trace.selected}`,
    `option_mass: ${trace.optionMass === undefined ? "n/a" : trace.optionMass.toFixed(4)}`,
    `probability_status: ${trace.probabilityStatus}`,
    `prompt_version: ${trace.promptVersion}`,
    `prompt_sha256: ${trace.promptSha256}`,
  ];

  if (evidence.length > 0) {
    lines.push(
      `retrieved_chunks: ${evidence
        .map((result) => `${result.chunk.id} (${result.score.toFixed(2)})`)
        .join(", ")}`,
    );
  }

  return lines.join("\n");
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Decision panel is missing ${selector}.`);
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
