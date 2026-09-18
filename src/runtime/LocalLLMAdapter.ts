import {
  DECISION_READOUT_TEMPERATURE,
  MAX_DECISION_OPTIONS,
  type LabelLogprob,
} from "./DecisionRuntime";

export type LocalLLMExecutionMethod = "logit" | "structured" | "generated" | "head";

export interface LocalLLMCapabilities {
  binary: boolean;
  choice: boolean;
  score: boolean;
  batch: boolean;
  distribution: boolean;
  executionMethods: LocalLLMExecutionMethod[];
}

export interface LocalLLMExecutionRequest {
  prompt: string;
  systemPrompt: string;
  candidateLabels: readonly string[];
  context: string;
}

export interface LocalLLMExecutionResult {
  candidates: LabelLogprob[];
  promptTokens?: number;
  executionMethod: LocalLLMExecutionMethod;
  model?: string;
  modelRevision?: string;
  browser?: string;
  webgpu?: boolean;
  gpu?: string;
  quantization?: string;
  webllmVersion?: string;
}

export interface LocalLLMAdapter {
  readonly capabilities: LocalLLMCapabilities;
  execute(request: LocalLLMExecutionRequest): Promise<LocalLLMExecutionResult>;
}

export const LOCAL_LLM_DECISION_CAPABILITIES: LocalLLMCapabilities = {
  binary: true,
  choice: true,
  score: true,
  batch: true,
  distribution: true,
  executionMethods: ["logit"],
};

export const LOCAL_LLM_READOUT_TEMPERATURE = DECISION_READOUT_TEMPERATURE;
export const LOCAL_LLM_MAX_CANDIDATE_LABELS = MAX_DECISION_OPTIONS;
