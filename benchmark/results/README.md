# Benchmark results

Exported artifacts from `npm run benchmark` live here, one JSON file per run.

Commit a result when it was produced by a **pinned** model on a real WebGPU
device, so a later run can be compared against it. The exporter names files

```
decision-benchmark-<model>-<timestamp>.json
```

and marks anything produced by the stub engine with `-stub`. A stub run says
`"engine": "stub"` in its environment block; it measures the benchmark itself,
never a model, so there is no reason to keep one here.

Before recording a result:

```bash
npm run pin:model          # writes the current Hugging Face commit shas
npm run pin:model -- --check
```

A run whose `model.pinned` is `false` records which weights it *asked* for, not
which it got. The report header says so in the same breath as the model name.

## What an artifact holds

| Section | Contents |
| --- | --- |
| `environment` | browser, platform, cores, memory, WebGPU adapter, engine, timestamp |
| `model` | model id, repo, revision, whether it is pinned, model library, web-llm version |
| `runtime` | prompt version, readout temperature, top-k, retrieval limit, low-mass threshold |
| `benchmark` | conditions, per-condition summaries, agreement per context source, duration |
| `results` | one row per fixture × condition, including per-row `promptSha256` |
| `probes` | option-mass probes: no correct answer, only the mass the labels held |

No result is committed yet. This directory holds the shape, not a claim.
