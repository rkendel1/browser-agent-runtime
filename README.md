# browser-agent-runtime

How much agent capability can we get out of a small browser-local model by changing the interface
between the model and the application?

Two runtimes, side by side:

```
ModelRuntime
  └── generate() → text

DecisionRuntime
  └── decide()   → typed probabilities
```

## Included pieces

- `WebLLMRuntime` backed by WebGPU via `@mlc-ai/web-llm`
- `WebLLMDecisionRuntime`: typed decisions read from the model's option-token logits
- lexical `ContextStore` with chunked page retrieval
- a single `search_context` tool
- `decideFromContext`: page → ContextStore → search_context → DecisionRuntime → typed decision
- observable `AgentLoop` trace
- demo UI with an entire-page vs retrieved-context toggle and a Browser Decision panel
- `benchmark/tasks.json` with 20 retrieval tasks, `benchmark/decisions.json` with 20 decisions
- a browser benchmark comparing a generated answer against a direct logit readout

## Decisions instead of answers

```ts
const result = await decisionRuntime.decide({
  state: "Customer cannot access their account after a password reset.",
  question: "Which queue should handle this request?",
  options: [
    { id: "access", description: "Account access support." },
    { id: "billing", description: "Billing support." },
  ],
});
```

```json
{
  "selected": "access",
  "probabilities": { "access": 0.91, "billing": 0.09 },
  "latencyMs": 742
}
```

No JSON is generated. No explanation is generated. No prose is parsed.

The prompt is fixed and ends at the answer position:

```
State:
{{state}}
Question:
{{question}}
Options:
A. {{option A}}
B. {{option B}}
Select the option that best answers the question.
```

The runtime asks web-llm for a single decode step with `logprobs: true`, keeps the tokens at that
position that name an option label, and softmaxes their logprobs into a distribution over the
option ids. The token the sampler would have emitted is discarded rather than parsed, which is why
the trace reports `generated_tokens: 0` — nothing the model wrote is read back.

The technique follows [OpenJev](https://github.com/rkendel1/open-jev)'s direct mode
(`src/openjev_phase1/direct.py`): declared options as single-letter answer slots, one forward pass,
softmax over the slot logits, no sampled answer token. Two differences are forced by the browser:

- **Top-k, not full vocabulary.** OpenJev indexes the full logit vector by the tokenizer's slot
  token ids. web-llm exposes at most the top 5 next-token candidates, so a readout separates at most
  5 options, and an option that never reaches the top-k reads as probability 0. `optionMass` in the
  trace (OpenJev's `allowed_token_mass`) records how much probability the option labels actually
  held; a low value means the model was not answering the question, however confident the
  normalized numbers look.
- **Slot matching by spelling.** OpenJev verifies each letter is one exact round-trip token. The
  browser runtime has no tokenizer handle, so it matches the returned token strings (`"A"`, `" A"`,
  `"A."`) and refuses to guess when no option label appears at all.

As in OpenJev, the probabilities are a conditional option score under this prompt. They are not
calibrated decision confidence.

## Context → Decision

```ts
const { decision, evidence } = await decideFromContext(contextStore, decisionRuntime, {
  query: "return policy 45 days",
  question: "Can this product be returned after 45 days?",
  options: [
    { id: "yes", description: "The product can be returned." },
    { id: "no", description: "The product cannot be returned." },
  ],
});
```

The whole path runs in the browser:

```
Web page → DOM extraction → ContextStore → relevant context
        → DecisionRuntime → WebGPU model → option probabilities → typed result
```

The demo's Browser Decision panel runs exactly this when "Use retrieved context from the page HTML"
is checked.

## Benchmark

`npm run benchmark` opens a page that runs `benchmark/decisions.json` (20 fixtures: 15 binary, 5
with 3–4 options) through both paths on **the same model instance**:

```
                SAME MODEL
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
    Generated answer     Direct logits
          ↓                   ↓
       parse choice        choice
```

It reports accuracy, median latency, output tokens and context tokens per path, plus how often the
two paths picked the same option:

| Path | Accuracy | Median latency | Output tokens |
| --- | --- | --- | --- |
| Generated | — | — | — |
| Direct logits | — | — | 0 |

The table is left empty on purpose. This PR's job is to establish the measurement, not to hit an
accuracy target. Agreement between the paths is a systems comparison, not a claim that the two
readouts are semantically equivalent — the same caution OpenJev's benchmark carries.

The generated arm is deliberately the most compact generation there is: one letter, no JSON, no
explanation. It is also the arm that cannot return a distribution — a parsed answer is one option at
probability 1.

### Pinning the model

`src/runtime/decisionModels.ts` is the one place that decides which weights the benchmark runs on.
web-llm's prebuilt catalogue points at the `main` branch of the weight repository, so a model id
alone does not pin anything, and OpenJev refuses a remote model without a 40-character commit
revision for exactly that reason.

Both entries currently ship with `revision: UNPINNED_REVISION` and the demo says so in the UI
(`@ main (UNPINNED — results are not reproducible)`). Before recording a result, pin them:

```bash
curl -s https://huggingface.co/api/models/mlc-ai/Qwen3.5-4B-q4f16_1-MLC \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"
```

and put the sha in `revision`. The runtime then loads weights from
`https://huggingface.co/{repo}/resolve/{revision}/` instead of the default branch. The WebGPU model
library URL and the `@mlc-ai/web-llm` version are already pinned exactly.

## Development

```bash
npm install
npm run test
npm run build
npm run dev        # http://localhost:5173/demo/index.html
npm run benchmark  # http://localhost:5173/demo/benchmark.html
```

Both demo pages need a WebGPU-capable browser; the first run downloads the model weights.

## Not Eve

This is `DecisionRuntime`, not Eve. Eve can eventually consume the primitive:

```
Eve
 ├── GenerateRuntime
 ├── DecisionRuntime
 ├── ToolRuntime
 └── EvidenceRuntime
```

Keeping the names apart keeps this repo an honest experiment.
