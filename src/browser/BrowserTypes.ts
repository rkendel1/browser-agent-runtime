import type { DecisionRuntime } from "../runtime/DecisionRuntime";

export type BrowserElementRole =
  | "button"
  | "link"
  | "input"
  | "select"
  | "checkbox"
  | "radio"
  | "heading"
  | "text"
  | "image"
  | "unknown";

export interface BrowserElement {
  id: string;
  role: BrowserElementRole;
  name?: string;
  value?: string;
  checked?: boolean;
  disabled?: boolean;
  bounds?: { x: number; y: number; width: number; height: number };
  visible: boolean;
  enabled: boolean;
}

export interface BrowserObservation {
  url: string;
  title?: string;
  viewport: { width: number; height: number };
  elements: BrowserElement[];
  /** Bounded rendered text. Page-derived text is always untrusted data. */
  text?: string;
  state?: Record<string, unknown>;
  timestamp: number;
  provenance: { source: "browser"; snapshotId: string };
}

export type BrowserAction =
  | { type: "click"; elementId: string }
  | { type: "type"; elementId: string; value: string }
  | { type: "select"; elementId: string; value: string }
  | { type: "check"; elementId: string }
  | { type: "uncheck"; elementId: string }
  | { type: "scroll"; direction: "up" | "down"; amount?: number }
  | { type: "navigate"; url: string }
  | { type: "wait"; ms: number }
  | { type: "stop" };

export interface BrowserDecisionOption {
  id: string;
  description: string;
  action: BrowserAction;
}

export interface BrowserDecisionRequest {
  observation: BrowserObservation;
  options: BrowserDecisionOption[];
  instruction: string;
  context?: Record<string, unknown>;
}

export type ConfidenceStatus = "available" | "unavailable" | "unsupported" | "invalid";

export interface BrowserDecision {
  optionId: string;
  confidence?: number;
  confidenceStatus: ConfidenceStatus;
  optionMass?: number;
  optionMassStatus: ConfidenceStatus;
  path: "direct" | "generated";
  model: { id: string; revision?: string; runtime: string };
  evidence: { snapshotId: string; contextTokens?: number; generatedTokens?: number };
  latencyMs: number;
}

export type BrowserBlockReason =
  | "stale_decision"
  | "unknown_option"
  | "missing_element"
  | "element_not_visible"
  | "element_disabled"
  | "invalid_role_action"
  | "invalid_navigation"
  | "missing_capability"
  | "invalid_decision"
  | "observation_limit"
  | "latency_limit"
  | "step_limit"
  | "retry_limit"
  | "execution_failed";

export interface BrowserExecutionResult {
  status: "success" | "complete" | "blocked" | "failed";
  durationMs: number;
  reason?: BrowserBlockReason;
  error?: string;
  resultingObservation?: BrowserObservation;
}

export interface BrowserActionEvidence {
  snapshotId: string;
  decision: {
    optionId: string;
    path: "direct" | "generated";
    latencyMs?: number;
    contextTokens?: number;
    generatedTokens?: number;
    model?: BrowserDecision["model"];
  };
  action: BrowserAction;
  execution: { status: "success" | "blocked" | "failed"; durationMs: number };
  resultingSnapshotId?: string;
}

export type BrowserContextMode = "state" | "page" | "retrieved";

export interface BrowserTask {
  instruction: string;
  options:
    | BrowserDecisionOption[]
    | ((observation: BrowserObservation, step: number) => BrowserDecisionOption[]);
  contextMode?: BrowserContextMode;
  context?: Record<string, unknown>;
  retrievedContext?: unknown;
  maxSteps?: number;
  maxLatency?: number;
  maxRetries?: number;
  maxObservationBytes?: number;
}

export interface BrowserRunResult {
  status: "complete" | "blocked" | "failed";
  reason?: BrowserBlockReason;
  steps: number;
  latencyMs: number;
  evidence: BrowserActionEvidence[];
  finalObservation?: BrowserObservation;
  error?: string;
}

/** The browser driver boundary. Methods are optional so capability checks are explicit. */
export interface BrowserExecutor {
  observe(): Promise<BrowserObservation>;
  click?(elementId: string): Promise<void>;
  type?(elementId: string, value: string): Promise<void>;
  select?(elementId: string, value: string): Promise<void>;
  check?(elementId: string): Promise<void>;
  uncheck?(elementId: string): Promise<void>;
  scroll?(direction: "up" | "down", amount?: number): Promise<void>;
  navigate?(url: string): Promise<void>;
  wait?(ms: number): Promise<void>;
}

export interface BrowserAgentRuntime {
  observe(): Promise<BrowserObservation>;
  decide(request: BrowserDecisionRequest): Promise<BrowserDecision>;
  execute(action: BrowserAction, observation: BrowserObservation): Promise<BrowserExecutionResult>;
  run(task: BrowserTask): Promise<BrowserRunResult>;
}

export interface BrowserDecisionRuntimeOptions {
  path?: "direct" | "generated";
}

/** Structural alias documenting that the browser reuses, rather than replaces, DecisionRuntime. */
export type BrowserDecisionBackend = DecisionRuntime;
