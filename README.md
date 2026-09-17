# browser-agent-runtime

Minimal MVP for proving that a small browser-local model becomes more useful when it retrieves only the page context it needs.

## Included MVP pieces

- `WebLLMRuntime` backed by WebGPU via `@mlc-ai/web-llm`
- lexical `ContextStore` with chunked page retrieval
- a single `search_context` tool
- observable `AgentLoop` trace
- demo UI with an entire-page vs retrieved-context toggle
- `benchmark/tasks.json` with 20 small benchmark tasks

## Development

```bash
npm install
npm run test
npm run build
npm run dev
```

Open `http://localhost:5173/demo/index.html` when running the demo locally.