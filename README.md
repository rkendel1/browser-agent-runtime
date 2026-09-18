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
- `DefaultBrowserAgentRuntime`: bounded observe → decide → validate → execute loop
- replaceable `BrowserExecutor` boundary and a browser-local `DOMBrowserExecutor`
- semantic browser observations, snapshot invalidation, explicit context modes, and action evidence
- lexical `ContextStore` with chunked page retrieval
- a single `search_context` tool
- `decideFromContext`: page → ContextStore → search_context → DecisionRuntime → typed decision
- observable `AgentLoop` trace
- demo UI with an entire-page vs retrieved-context toggle and a Browser Decision panel
- `benchmark/tasks.json` with 20 retrieval tasks, `benchmark/decisions.json` with 20 decisions
  (each carrying a page and a retrieval query) and six option-mass probes
- a browser benchmark that runs both readouts across three context sources on one
  loaded model, and exports the whole run as JSON

## Browser execution runtime

The model chooses only from capabilities declared by the application. The runtime resolves the
chosen option back to its typed action, checks it against a fresh browser snapshot and policy, and
only then calls the browser adapter.

```ts
const browser = new DOMBrowserExecutor();
const runtime = new DefaultBrowserAgentRuntime(browser, decisionRuntime);

const result = await runtime.run({
  instruction: "Continue checkout if the order is ready.",
  contextMode: "state",
  maxSteps: 5,
  maxLatency: 10_000,
  maxRetries: 1,
  maxObservationBytes: 64_000,
  options: [
    {
      id: "continue",
      description: "Continue to payment",
      action: { type: "click", elementId: "checkout-button" },
    },
    { id: "stop", description: "Stop", action: { type: "stop" } },
  ],
});
```

`state` excludes rendered page text, `page` adds bounded page text, and `retrieved` adds only
explicitly supplied retrieved context. Page-derived content is labelled untrusted data. It never
changes the task, option set, authorization, action schema, or navigation policy. Snapshot IDs are
checked immediately before execution, so a decision from snapshot A is blocked if the page has
already advanced to snapshot B.

Direct and generated decision runtimes both normalize to `BrowserDecision`; browser execution has
one implementation. Generated readouts mark probability diagnostics `unsupported`, and adapters
can do the same for model families that cannot expose valid option mass—unsupported extraction is
never represented as zero.

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

`npm run benchmark` opens a page that runs every condition on **one loaded
model** in one browser: the same weights, the same fixtures, the same session.

```
                        Chromium
                           ↓
                        WebGPU
                           ↓
                       MLCEngine
                           ↓
                   same model weights
                    ├── GeneratedDecisionRuntime
                    └── WebLLMDecisionRuntime
```

### The experiment

Each of the 20 fixtures carries a page that contains its state among distractor
sections, so context quality is a variable rather than an assumption. Every
fixture runs through two readouts across three context sources:

|                   | Generated | Direct logits |
| --- | --- | --- |
| Full page         | A | C |
| Retrieved context | B | D |
| State only        | control | control |

The state-only row is the control: the decision with no page and no retrieval in
the way. Retrieved-context rows also record whether the retrieved chunks
actually held the state (`retrievedStateHit`), so a wrong answer over context
that never contained the evidence is not charged to the decision path.

Nothing here predicts a winner. The table below is empty because no run on real
weights has happened yet.

| Path | Accuracy | Median latency | Output tokens |
| --- | --- | --- | --- |
| Generated | — | — | — |
| Direct logits | — | — | 0 |

### What the two paths report

They are not the same quantity, and the report keeps them apart:

```
Direct logits          Generated
  selected               selected
  optionMass             latency
  latency                generatedTokens
              Agreement: yes/no
```

The direct path reports P(option label | prompt). The generated path reports a
parsed choice. Agreement between them is a systems comparison, not evidence that
the two readouts are semantically equivalent — the same caution OpenJev's
benchmark carries. The direct probabilities are never called confidence.

### Option mass, and the failure it catches

`optionMass` is the probability the model left on the option labels before
renormalization (OpenJev's `allowed_token_mass`). A decision can read

```json
{
  "selected": "yes",
  "probabilities": { "yes": 0.51, "no": 0.49 },
  "optionMass": 0.18,
  "lowOptionMass": true
}
```

which is not "51% confidence". It is a near-tie between two labels that together
held under a fifth of the model's next-token probability — the model was not
answering the question. `benchmark/option-mass-probes.json` holds six decisions
built to provoke exactly this: unrelated evidence, no evidence, both options
true, neither option applicable, a question that invites prose, and
self-contradicting evidence. They are scored on nothing; the run only records
what the mass did.

`lowOptionMass` is a diagnostic marker against a provisional threshold
(`LOW_OPTION_MASS_THRESHOLD`, currently 0.5). **No runtime behaves differently
when it is set.** Deciding what an agent should do about weak support is a later
question; this measures how often it happens first.

### Readout temperature

The direct readout runs at `temperature: 1`, and that is load-bearing. web-llm
builds the reported `top_logprobs` from `softmax(logits / max(temperature,
1e-6))` — the distribution it also samples from. At `temperature: 0` that clamp
returns a one-hot vector: every decision reads 100% against 0%, and option mass
degenerates into "was the argmax a label". At temperature 1 the numbers are the
model's own next-token distribution. The sampled token is still discarded, so
nothing about the decision is random.

### Reports and artifacts

A run produces a readable report and an **Export JSON** artifact:

```json
{
  "environment": { "userAgent": "…", "gpu": {}, "engine": "webgpu" },
  "model": { "modelId": "…", "revision": "…", "pinned": true },
  "runtime": { "promptVersion": "…", "readoutTemperature": 1 },
  "benchmark": { "summaries": [], "agreements": [] },
  "results": [],
  "probes": []
}
```

Every row carries its own `promptSha256`, so a result can prove which prompt
produced it. Artifacts belong in `benchmark/results/` — see the README there.

### Pinning the model

`src/runtime/decisionModels.ts` decides which weights the benchmark runs on.
web-llm's prebuilt catalogue resolves weights from the repository's default
branch, so a model id alone pins nothing, and OpenJev refuses a remote model
without a 40-character commit revision for exactly that reason.

```bash
npm run pin:model            # write the current commit sha for each model
npm run pin:model -- --check # fail if anything is still unpinned
```

Until that runs, the demo and the report say
`@ main (UNPINNED — results are not reproducible)` and the artifact records
`"pinned": false`. The model library URL and the `@mlc-ai/web-llm` version are
already pinned exactly.

The benchmark model is `Qwen2.5-1.5B-Instruct-q4f16_1-MLC`: ~1.6 GB of VRAM,
loads reliably, and is not a thinking model, so the generated arm produces an
answer rather than a reasoning trace. `Qwen3.5-4B-q4f16_1-MLC` is offered as a
second data point. Neither is a claim about which model is best for the job.

### Testing the benchmark itself

`/demo/benchmark.html?engine=stub` swaps the model for a deterministic fake
(`demo/stubEngine.ts`) so the conditions, retrieval, report and export can be
exercised on a machine that cannot download weights. Such a run is stamped
`"engine": "stub"` and the page says in red that every number is fabricated. It
tests the apparatus; it is never a result about a model.

## Development

```bash
npm install
npm run test
npm run build
npm run dev          # http://localhost:5173/demo/index.html
npm run benchmark    # http://localhost:5173/demo/benchmark.html
npm run pin:model    # pin the model weights before recording a result
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
