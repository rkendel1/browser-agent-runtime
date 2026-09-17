import { AgentLoop, type AgentTraceEntry, type ContextMode, WebLLMRuntime, getPage } from "../src";
import benchmarkTasksData from "../benchmark/tasks.json";

interface BenchmarkTask {
  id: string;
  name: string;
  goal: string;
  expected: string;
  pageHtml: string;
}

const taskSelect = document.querySelector<HTMLSelectElement>("#task-select");
const goalInput = document.querySelector<HTMLTextAreaElement>("#goal-input");
const modelInput = document.querySelector<HTMLInputElement>("#model-input");
const pageInput = document.querySelector<HTMLTextAreaElement>("#page-input");
const runButton = document.querySelector<HTMLButtonElement>("#run-button");
const status = document.querySelector<HTMLParagraphElement>("#status");
const traceContainer = document.querySelector<HTMLDivElement>("#trace");
const modelStat = document.querySelector<HTMLDivElement>("#model-stat");
const contextStat = document.querySelector<HTMLDivElement>("#context-stat");
const tokenStat = document.querySelector<HTMLDivElement>("#token-stat");

const benchmarkTasks = benchmarkTasksData as BenchmarkTask[];

void initialize();

async function initialize(): Promise<void> {
  if (
    !taskSelect ||
    !goalInput ||
    !modelInput ||
    !pageInput ||
    !runButton ||
    !status ||
    !traceContainer ||
    !modelStat ||
    !contextStat ||
    !tokenStat
  ) {
    throw new Error("Demo UI failed to initialize.");
  }

  taskSelect.innerHTML = benchmarkTasks
    .map((task) => `<option value="${task.id}">${task.name}</option>`)
    .join("");

  taskSelect.addEventListener("change", () => loadTask(taskSelect.value));
  runButton.addEventListener("click", () => {
    void runTask();
  });

  loadTask(benchmarkTasks[0]?.id);
}

function loadTask(taskId: string | undefined): void {
  const task = benchmarkTasks.find((candidate) => candidate.id === taskId) ?? benchmarkTasks[0];
  if (!task || !goalInput || !pageInput) {
    return;
  }

  goalInput.value = task.goal;
  pageInput.value = task.pageHtml;
  renderTrace([]);
  if (status) {
    status.textContent = `Expected: ${task.expected}`;
  }
}

async function runTask(): Promise<void> {
  if (
    !goalInput ||
    !modelInput ||
    !pageInput ||
    !status ||
    !traceContainer ||
    !modelStat ||
    !contextStat ||
    !tokenStat
  ) {
    return;
  }

  const mode =
    document.querySelector<HTMLInputElement>('input[name="context-mode"]:checked')?.value === "entire-page"
      ? "entire-page"
      : "retrieved-context";

  status.textContent = "Loading model and running task…";
  renderTrace([]);

  try {
    const runtime = new WebLLMRuntime({
      model: modelInput.value.trim(),
      initProgressCallback: (progress) => {
        status.textContent = progress.text ?? "Loading model…";
      },
    });
    const agent = new AgentLoop(runtime);
    const document = new DOMParser().parseFromString(pageInput.value, "text/html");
    const page = getPage(document);
    const result = await agent.run({
      goal: goalInput.value.trim(),
      page,
      mode: mode as ContextMode,
    });

    modelStat.textContent = modelInput.value.trim();
    contextStat.textContent = mode === "entire-page" ? "Entire page" : "Retrieved context";
    tokenStat.textContent = `${result.contextTokens.toLocaleString()} tokens`;
    status.textContent = result.answer;
    renderTrace(result.trace);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Unknown error";
    traceContainer.innerHTML = "";
  }
}

function renderTrace(trace: AgentTraceEntry[]): void {
  if (!traceContainer) {
    return;
  }

  traceContainer.innerHTML =
    trace.length === 0
      ? `<div class="trace-step"><strong>READY</strong><div>Select a task and run the agent.</div></div>`
      : trace
          .map(
            (entry) => `
              <article class="trace-step">
                <strong>${entry.type.toUpperCase()}</strong>
                <div>${escapeHtml(entry.content).replace(/\n/g, "<br />")}</div>
              </article>
            `,
          )
          .join("");
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
