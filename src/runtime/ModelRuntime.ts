export type MessageRole = "user" | "assistant";

export interface Message {
  role: MessageRole;
  content: string;
}

export interface ModelRequest {
  system: string;
  messages: Message[];
}

export interface ModelRuntime {
  generate(request: ModelRequest): Promise<string>;
}

