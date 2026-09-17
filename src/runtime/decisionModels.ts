/**
 * Pinned models for the decision benchmark.
 *
 * OpenJev refuses to load a remote model without a 40-character commit
 * revision, because a benchmark that names only `main` is not reproducible. The
 * browser has the same problem: web-llm's prebuilt config points at the `main`
 * branch of the weight repository, so the weights behind a model id can change
 * under a saved result.
 *
 * This module is the one place that decides which weights the benchmark runs
 * on. Fill in `revision` with the commit you measured:
 *
 *   curl -s https://huggingface.co/api/models/mlc-ai/Qwen3.5-4B-q4f16_1-MLC \
 *     | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"
 */

export interface PinnedDecisionModel {
  /** web-llm model id; also the cache key for downloaded weights. */
  modelId: string;
  /** Hugging Face repository holding the MLC-converted weights. */
  repo: string;
  /**
   * A 40-character commit sha, or `UNPINNED_REVISION` when the weights are
   * being taken from the repository's default branch.
   */
  revision: string;
  /** Exact WebGPU model library. The `v0_2_84` path segment is the build pin. */
  modelLib: string;
  vramRequiredMB: number;
  contextWindowSize: number;
}

/** The value `revision` carries when the weights are not pinned to a commit. */
export const UNPINNED_REVISION = "main";

/** web-llm release the pinned model libraries below were read from. */
export const WEB_LLM_VERSION = "0.2.85";

/**
 * The benchmark model: small enough that web-llm loads it reliably on ordinary
 * laptops (~1.6 GB of VRAM), and not a thinking model, so the generated arm is
 * a plain answer rather than a reasoning trace.
 *
 * This is not a claim that it is the best model for the job. Model selection is
 * a later question; this one exists so the measurement can happen at all.
 */
export const BENCHMARK_DECISION_MODEL: PinnedDecisionModel = {
  modelId: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
  repo: "mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
  revision: UNPINNED_REVISION,
  modelLib:
    "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/Qwen2-1.5B-Instruct-q4f16_1_cs1k-webgpu.wasm",
  vramRequiredMB: 1629.75,
  contextWindowSize: 4096,
};

/**
 * Qwen3.5-4B, the family OpenJev pins for its own benchmark, for machines with
 * ~3.9 GB of VRAM to spare. Offered for a second data point, not as the
 * default.
 */
export const LARGE_DECISION_MODEL: PinnedDecisionModel = {
  modelId: "Qwen3.5-4B-q4f16_1-MLC",
  repo: "mlc-ai/Qwen3.5-4B-q4f16_1-MLC",
  revision: UNPINNED_REVISION,
  modelLib:
    "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm",
  vramRequiredMB: 3867.82,
  contextWindowSize: 4096,
};

export const DECISION_MODELS: PinnedDecisionModel[] = [
  BENCHMARK_DECISION_MODEL,
  LARGE_DECISION_MODEL,
];

export function isPinnedRevision(revision: string): boolean {
  return /^[0-9a-f]{40}$/.test(revision);
}

export function findDecisionModel(modelId: string): PinnedDecisionModel | undefined {
  return DECISION_MODELS.find((model) => model.modelId === modelId);
}

/**
 * Hugging Face URL in the `resolve/{revision}` form web-llm accepts, so a
 * pinned revision reaches the loader instead of only the documentation.
 */
export function modelWeightsUrl(model: PinnedDecisionModel): string {
  return `https://huggingface.co/${model.repo}/resolve/${model.revision}/`;
}

/** Human-readable provenance line for a trace or a benchmark report. */
export function describeModelPin(model: PinnedDecisionModel): string {
  return isPinnedRevision(model.revision)
    ? `${model.modelId} @ ${model.revision.slice(0, 12)} (pinned)`
    : `${model.modelId} @ ${model.revision} (UNPINNED — results are not reproducible)`;
}

export interface WebLLMAppConfig {
  model_list: Array<{
    model: string;
    model_id: string;
    model_lib: string;
    vram_required_MB?: number;
    overrides?: { context_window_size?: number };
  }>;
}

/**
 * An `appConfig` that resolves the given models from their pinned revisions
 * rather than from web-llm's prebuilt list.
 */
export function decisionAppConfig(models: PinnedDecisionModel[] = DECISION_MODELS): WebLLMAppConfig {
  return {
    model_list: models.map((model) => ({
      model: modelWeightsUrl(model),
      model_id: model.modelId,
      model_lib: model.modelLib,
      vram_required_MB: model.vramRequiredMB,
      overrides: { context_window_size: model.contextWindowSize },
    })),
  };
}
