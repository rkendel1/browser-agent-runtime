import type {
  BrowserElement,
  BrowserElementRole,
  BrowserExecutor,
  BrowserObservation,
} from "./BrowserTypes";

export interface DOMBrowserExecutorOptions {
  document?: Document;
  window?: Window;
  maxElements?: number;
  maxTextCharacters?: number;
}

/** Browser-local adapter. It exposes a bounded semantic snapshot, never HTML or the raw DOM. */
export class DOMBrowserExecutor implements BrowserExecutor {
  private readonly document: Document;
  private readonly window: Window;
  private readonly maxElements: number;
  private readonly maxTextCharacters: number;
  private readonly generatedIds = new WeakMap<Element, string>();
  private readonly elementsById = new Map<string, Element>();
  private nextId = 1;

  constructor(options: DOMBrowserExecutorOptions = {}) {
    const documentRef = options.document ?? globalThis.document;
    const windowRef = options.window ?? globalThis.window;
    if (!documentRef || !windowRef) throw new Error("DOMBrowserExecutor requires a browser document.");
    this.document = documentRef;
    this.window = windowRef;
    this.maxElements = options.maxElements ?? 200;
    this.maxTextCharacters = options.maxTextCharacters ?? 32_000;
  }

  async observe(): Promise<BrowserObservation> {
    this.elementsById.clear();
    const candidates = Array.from(
      this.document.querySelectorAll(
        "button,a[href],input,textarea,select,[role],h1,h2,h3,h4,h5,h6,img,p,li",
      ),
    ).slice(0, this.maxElements);
    const elements = candidates.map((element) => this.toBrowserElement(element));
    const text = (this.document.body?.innerText ?? this.document.body?.textContent ?? "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, this.maxTextCharacters);
    const snapshotMaterial = JSON.stringify({
      url: this.window.location.href,
      title: this.document.title,
      viewport: [this.window.innerWidth, this.window.innerHeight],
      scroll: [this.window.scrollX, this.window.scrollY],
      elements,
      text,
    });
    return {
      url: this.window.location.href,
      title: this.document.title || undefined,
      viewport: { width: this.window.innerWidth, height: this.window.innerHeight },
      elements,
      text,
      timestamp: Date.now(),
      provenance: { source: "browser", snapshotId: `dom-${fnv1a(snapshotMaterial)}` },
    };
  }

  async click(elementId: string): Promise<void> {
    this.element(elementId).dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  }

  async type(elementId: string, value: string): Promise<void> {
    const element = this.element(elementId);
    if (!(element instanceof HTMLInputElement) && !(element instanceof HTMLTextAreaElement)) {
      throw new Error(`Element ${elementId} is not text-editable.`);
    }
    element.focus();
    element.value = value;
    element.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }

  async select(elementId: string, value: string): Promise<void> {
    const element = this.element(elementId);
    if (!(element instanceof HTMLSelectElement)) throw new Error(`Element ${elementId} is not a select.`);
    if (!Array.from(element.options).some((option) => option.value === value)) {
      throw new Error(`Select ${elementId} has no option ${JSON.stringify(value)}.`);
    }
    element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }

  async check(elementId: string): Promise<void> {
    this.setChecked(elementId, true);
  }

  async uncheck(elementId: string): Promise<void> {
    this.setChecked(elementId, false);
  }

  async scroll(direction: "up" | "down", amount = Math.round(this.window.innerHeight * 0.8)) {
    this.window.scrollBy({ top: direction === "down" ? amount : -amount, behavior: "instant" });
  }

  async navigate(url: string): Promise<void> {
    this.window.location.assign(url);
  }

  async wait(ms: number): Promise<void> {
    await new Promise<void>((resolve) => globalThis.setTimeout(resolve, ms));
  }

  private toBrowserElement(element: Element): BrowserElement {
    const id = this.elementId(element);
    this.elementsById.set(id, element);
    const role = semanticRole(element);
    const rect = element.getBoundingClientRect();
    const style = this.window.getComputedStyle(element);
    const visible =
      rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
    const formElement = element as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
    const disabled = "disabled" in formElement && Boolean(formElement.disabled);
    const browserElement: BrowserElement = {
      id,
      role,
      name: accessibleName(element) || undefined,
      visible,
      enabled: visible && !disabled,
      disabled,
      bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
    };
    if ("value" in formElement) browserElement.value = String(formElement.value);
    if (element instanceof HTMLInputElement && (element.type === "checkbox" || element.type === "radio")) {
      browserElement.checked = element.checked;
    }
    return browserElement;
  }

  private elementId(element: Element): string {
    const explicit = element.getAttribute("data-browser-agent-id") || element.id;
    if (explicit) return explicit;
    let generated = this.generatedIds.get(element);
    if (!generated) {
      generated = `element-${this.nextId}`;
      this.nextId += 1;
      this.generatedIds.set(element, generated);
    }
    return generated;
  }

  private element(id: string): Element {
    const element = this.elementsById.get(id);
    if (!element || !element.isConnected) throw new Error(`Element ${id} is not in the current snapshot.`);
    return element;
  }

  private setChecked(elementId: string, checked: boolean): void {
    const element = this.element(elementId);
    if (!(element instanceof HTMLInputElement) || !["checkbox", "radio"].includes(element.type)) {
      throw new Error(`Element ${elementId} is not checkable.`);
    }
    if (!checked && element.type === "radio") throw new Error("Radio elements cannot be unchecked directly.");
    element.checked = checked;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

function semanticRole(element: Element): BrowserElementRole {
  const explicit = element.getAttribute("role");
  if (explicit && isBrowserRole(explicit)) return explicit;
  const tag = element.tagName.toLowerCase();
  if (tag === "button") return "button";
  if (tag === "a") return "link";
  if (tag === "select") return "select";
  if (tag === "img") return "image";
  if (/^h[1-6]$/.test(tag)) return "heading";
  if (tag === "input") {
    const type = (element as HTMLInputElement).type;
    if (type === "checkbox") return "checkbox";
    if (type === "radio") return "radio";
    if (["button", "submit", "reset"].includes(type)) return "button";
    return "input";
  }
  if (tag === "textarea") return "input";
  if (["p", "li"].includes(tag)) return "text";
  return "unknown";
}

function isBrowserRole(role: string): role is BrowserElementRole {
  return ["button", "link", "input", "select", "checkbox", "radio", "heading", "text", "image", "unknown"].includes(role);
}

function accessibleName(element: Element): string {
  const labelledBy = element.getAttribute("aria-labelledby");
  const referenced = labelledBy
    ?.split(/\s+/)
    .map((id) => element.ownerDocument.getElementById(id)?.textContent?.trim())
    .filter(Boolean)
    .join(" ");
  return (
    referenced ||
    element.getAttribute("aria-label") ||
    element.getAttribute("alt") ||
    element.getAttribute("title") ||
    (element as HTMLInputElement).placeholder ||
    element.textContent?.replace(/\s+/g, " ").trim() ||
    ""
  );
}

function fnv1a(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
