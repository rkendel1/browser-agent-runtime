import type {
  BrowserAction,
  BrowserBlockReason,
  BrowserDecision,
  BrowserDecisionOption,
  BrowserDecisionRequest,
  BrowserElement,
  BrowserExecutor,
  BrowserObservation,
} from "./BrowserTypes";

export interface BrowserExecutionPolicy {
  allowNavigation?: (target: URL, observation: BrowserObservation) => boolean;
  maxWaitMs?: number;
}

export type BrowserValidationResult =
  | { ok: true; action: BrowserAction; option: BrowserDecisionOption }
  | { ok: false; reason: BrowserBlockReason; detail?: string };

export function validateDecision(
  decision: BrowserDecision,
  request: BrowserDecisionRequest,
  observation: BrowserObservation,
  executor?: BrowserExecutor,
  policy: BrowserExecutionPolicy = {},
): BrowserValidationResult {
  const option = request.options.find((candidate) => candidate.id === decision.optionId);
  if (!option) return blocked("unknown_option", decision.optionId);
  if (decision.evidence.snapshotId !== request.observation.provenance.snapshotId) {
    return blocked("invalid_decision", "Decision provenance does not match its request.");
  }
  const actionValidation = validateAction(
    option.action,
    request.observation,
    observation,
    executor,
    policy,
  );
  return actionValidation.ok ? { ok: true, action: option.action, option } : actionValidation;
}

export function validateAction(
  action: BrowserAction,
  source: BrowserObservation,
  current: BrowserObservation,
  executor?: BrowserExecutor,
  policy: BrowserExecutionPolicy = {},
): { ok: true } | { ok: false; reason: BrowserBlockReason; detail?: string } {
  if (source.provenance.snapshotId !== current.provenance.snapshotId) {
    return blocked("stale_decision", "The browser snapshot changed before execution.");
  }

  if (!hasCapability(action, executor)) {
    return blocked("missing_capability", action.type);
  }

  if (action.type === "navigate") {
    const target = safeUrl(action.url, current.url);
    if (!target || !isNavigationAllowed(target, current, policy)) {
      return blocked("invalid_navigation", action.url);
    }
    return { ok: true };
  }

  if (action.type === "wait") {
    const maxWaitMs = policy.maxWaitMs ?? 30_000;
    return Number.isSafeInteger(action.ms) && action.ms >= 0 && action.ms <= maxWaitMs
      ? { ok: true }
      : blocked("invalid_decision", `wait must be between 0 and ${maxWaitMs}ms`);
  }

  if (action.type === "scroll") {
    return action.amount === undefined || (Number.isFinite(action.amount) && action.amount > 0)
      ? { ok: true }
      : blocked("invalid_decision", "scroll amount must be positive");
  }
  if (action.type === "stop") return { ok: true };

  const sourceElement = source.elements.find((element) => element.id === action.elementId);
  const element = current.elements.find((candidate) => candidate.id === action.elementId);
  if (!sourceElement || !element) return blocked("missing_element", action.elementId);
  if (!element.visible) return blocked("element_not_visible", action.elementId);
  if (!element.enabled || element.disabled) return blocked("element_disabled", action.elementId);
  if (!validRole(action, element)) {
    return blocked("invalid_role_action", `${action.type} cannot target ${element.role}`);
  }
  return { ok: true };
}

function validRole(action: BrowserAction, element: BrowserElement): boolean {
  switch (action.type) {
    case "click":
      return ["button", "link", "input", "select", "checkbox", "radio"].includes(element.role);
    case "type":
      return element.role === "input";
    case "select":
      return element.role === "select";
    case "check":
      return (element.role === "checkbox" || element.role === "radio") && element.checked !== true;
    case "uncheck":
      return element.role === "checkbox" && element.checked === true;
    default:
      return true;
  }
}

function hasCapability(action: BrowserAction, executor: BrowserExecutor | undefined): boolean {
  if (!executor || action.type === "stop") return true;
  const method = executor[action.type];
  return typeof method === "function";
}

function isNavigationAllowed(
  target: URL,
  observation: BrowserObservation,
  policy: BrowserExecutionPolicy,
): boolean {
  if (target.protocol !== "http:" && target.protocol !== "https:") return false;
  if (policy.allowNavigation) return policy.allowNavigation(target, observation);
  const current = safeUrl(observation.url);
  return Boolean(current && target.origin === current.origin);
}

function safeUrl(value: string, base?: string): URL | undefined {
  try {
    return base ? new URL(value, base) : new URL(value);
  } catch {
    return undefined;
  }
}

function blocked(reason: BrowserBlockReason, detail?: string) {
  return { ok: false as const, reason, detail };
}
