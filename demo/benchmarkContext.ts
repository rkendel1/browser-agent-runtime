import DOMPurify from "dompurify";

import {
  ContextStore,
  getPage,
  searchContext,
  type BenchmarkEnvironment,
  type DecisionContextSource,
  type DecisionFixture,
  type PageSnapshot,
  type ResolvedState,
  type StateResolver,
} from "../src";

/** Browser, GPU and WebGPU adapter facts, recorded with every run. */
export async function captureEnvironment(
  engine: "webgpu" | "stub",
): Promise<BenchmarkEnvironment> {
  interface AdapterInfo {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
  }

  // `info` is the current spec; `requestAdapterInfo()` is the older shape some
  // browsers still ship. Both are read through a loose type because the DOM lib
  // version decides which one it knows about.
  interface LooseAdapter {
    info?: AdapterInfo;
    requestAdapterInfo?: () => Promise<AdapterInfo>;
  }

  const navigatorWithExtras = navigator as Navigator & {
    deviceMemory?: number;
    gpu?: { requestAdapter(): Promise<unknown> };
  };

  const environment: BenchmarkEnvironment = {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemoryGb: navigatorWithExtras.deviceMemory,
    webgpu: Boolean(navigatorWithExtras.gpu),
    engine,
    timestamp: new Date().toISOString(),
  };

  try {
    const adapter = (await navigatorWithExtras.gpu?.requestAdapter()) as LooseAdapter | null;
    const info = adapter?.info ?? (await adapter?.requestAdapterInfo?.());
    if (info) {
      environment.gpu = {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      };
    }
  } catch {
    // An adapter that refuses to describe itself is worth recording as absent
    // rather than failing the run.
  }

  return environment;
}

export interface FixtureContext {
  page: PageSnapshot;
  store: ContextStore;
}

/**
 * Resolves the state for each cell of the experiment.
 *
 *   state       the fixture's own statement, no page, no retrieval (control)
 *   full-page   every word of the fixture's page
 *   retrieved   what search_context returns for the fixture's query
 */
export function createStateResolver(retrievalLimit: number): StateResolver {
  const cache = new Map<string, FixtureContext>();

  const contextFor = (fixture: DecisionFixture): FixtureContext => {
    const cached = cache.get(fixture.id);
    if (cached) {
      return cached;
    }

    const sanitized = DOMPurify.sanitize(fixture.pageHtml, { WHOLE_DOCUMENT: true });
    const page = getPage(new DOMParser().parseFromString(sanitized, "text/html"));
    const store = new ContextStore();
    store.add(page.chunks);

    const prepared = { page, store };
    cache.set(fixture.id, prepared);
    return prepared;
  };

  return (fixture, context: DecisionContextSource): ResolvedState => {
    if (context === "state") {
      return { state: fixture.state };
    }

    const { page, store } = contextFor(fixture);

    if (context === "full-page") {
      return { state: page.fullText };
    }

    const retrieved = searchContext(store, fixture.query, retrievalLimit);
    const state = retrieved.results.map((result) => result.chunk.text).join("\n");

    return {
      state,
      retrievedChunkIds: retrieved.results.map((result) => result.chunk.id),
      // Did retrieval actually carry the evidence? Without this, a wrong answer
      // over context that never held the state looks like a decision failure.
      retrievedStateHit: normalize(state).includes(normalize(fixture.state)),
    };
  };
}

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
