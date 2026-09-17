import type { ModelRequest, ModelRuntime } from "./ModelRuntime";

interface WebLLMCompletion {
  choices?: Array<{
    message?: {
      content?: string | Array<{ text?: string }>;
    };
  }>;
}

interface WebLLMEngine {
  reload(model: string): Promise<void>;
  chat: {
    completions: {
      create(request: {
        messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
        temperature?: number;
      }): Promise<WebLLMCompletion>;
    };
  };
}

export interface WebLLMRuntimeOptions {
  model: string;
  temperature?: number;
  initProgressCallback?: (progress: { progress?: number; text?: string }) => void;
  createEngine?: () => Promise<WebLLMEngine>;
}

export class WebLLMRuntime implements ModelRuntime {
  private enginePromise?: Promise<WebLLMEngine>;
  private readonly options: WebLLMRuntimeOptions;

  constructor(options: WebLLMRuntimeOptions) {
    this.options = options;
  }

  async generate(request: ModelRequest): Promise<string> {
    const engine = await this.getEngine();
    const response = await engine.chat.completions.create({
      messages: [
        { role: "system", content: request.system },
        ...request.messages.map((message) => ({
          role: message.role,
          content: message.content,
        })),
      ],
      temperature: this.options.temperature ?? 0,
    });

    return normalizeContent(response.choices?.[0]?.message?.content);
  }

  private getEngine(): Promise<WebLLMEngine> {
    if (!this.enginePromise) {
      this.enginePromise = this.options.createEngine?.() ?? this.createDefaultEngine();
    }

    return this.enginePromise;
  }

  private async createDefaultEngine(): Promise<WebLLMEngine> {
    const webllm = (await import("@mlc-ai/web-llm")) as {
      MLCEngine: new (config?: unknown) => WebLLMEngine;
    };
    const engine = new webllm.MLCEngine({
      initProgressCallback: this.options.initProgressCallback,
    });
    await engine.reload(this.options.model);
    return engine;
  }
}

function normalizeContent(content: string | Array<{ text?: string }> | undefined): string {
  if (typeof content === "string") {
    return content.trim();
  }

  if (!content) {
    return "";
  }

  return content
    .map((part) => part.text?.trim())
    .filter((part): part is string => Boolean(part))
    .join("\n")
    .trim();
}
