/**
 * A deterministic stand-in for `MLCEngine`.
 *
 * It exists to exercise the benchmark apparatus — conditions, retrieval, the
 * report, the export — on machines that cannot download model weights. It is
 * NOT a model: every number it returns is made up by the functions below.
 *
 * Runs that use it are stamped `environment.engine: "stub"` and the UI says so
 * in red. Nothing it produces is a result about a model.
 */

interface StubRequest {
  messages: Array<{ role: string; content: string }>;
  logprobs?: boolean;
  top_logprobs?: number;
}

export interface StubEngine {
  reload(model: string): Promise<void>;
  getGPUVendor?(): Promise<string>;
  chat: {
    completions: {
      create(request: StubRequest): Promise<unknown>;
    };
  };
}

export function createStubEngine(): StubEngine {
  return {
    reload: async () => undefined,
    getGPUVendor: async () => "stub",
    chat: {
      completions: {
        create: async (request: StubRequest) => {
          const prompt = request.messages.map((message) => message.content).join("\n");
          const decision = scorePrompt(prompt);

          if (request.logprobs) {
            return {
              choices: [
                {
                  logprobs: {
                    content: [
                      {
                        token: decision.labels[decision.winner] ?? "A",
                        logprob: Math.log(Math.max(decision.probabilities[decision.winner] ?? 0.5, 1e-6)),
                        top_logprobs: buildTopLogprobs(decision, request.top_logprobs ?? 5),
                      },
                    ],
                  },
                },
              ],
              usage: { prompt_tokens: Math.ceil(prompt.length / 4) },
            };
          }

          return {
            choices: [{ message: { content: generatedAnswer(decision) } }],
          };
        },
      },
    },
  };
}

interface StubDecision {
  labels: string[];
  probabilities: number[];
  winner: number;
  /** Share of the fake next-token mass sitting on option labels. */
  optionMass: number;
  hash: number;
}

/**
 * Score the options by how much their wording overlaps the state, so the stub
 * is not uniformly random and the report has something to show. This is a
 * lexical toy, not inference.
 */
function scorePrompt(prompt: string): StubDecision {
  const state = section(prompt, "State:", "Question:");
  const optionsBlock = section(prompt, "Options:", "Select the option");
  const optionLines = optionsBlock
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^[A-H][.]/.test(line));

  const labels = optionLines.map((line) => line[0]!);
  const stateTerms = new Set(terms(state));
  const scores = optionLines.map((line) => {
    const overlap = terms(line.slice(2)).filter((term) => stateTerms.has(term)).length;
    return overlap;
  });

  const hash = hashString(prompt);
  // A little deterministic noise so the two stub paths do not agree perfectly.
  const jittered = scores.map((score, index) => score + ((hash >> (index * 3)) & 3) * 0.25);
  const probabilities = softmax(jittered);
  const winner = probabilities.reduce(
    (best, probability, index) => (probability > probabilities[best]! ? index : best),
    0,
  );

  return {
    labels,
    probabilities,
    winner,
    // Spread across [0.15, 0.99] so some rows trip the low-mass marker.
    optionMass: 0.15 + ((hash >>> 8) % 85) / 100,
    hash,
  };
}

function buildTopLogprobs(
  decision: StubDecision,
  topLogprobs: number,
): Array<{ token: string; logprob: number }> {
  const entries = decision.labels.map((label, index) => ({
    token: label,
    logprob: Math.log(Math.max(decision.probabilities[index]! * decision.optionMass, 1e-6)),
  }));

  // Whatever mass is not on a label shows up as a plausible non-label token,
  // which is what pushes optionMass below 1 in a real readout.
  entries.push({ token: "The", logprob: Math.log(Math.max(1 - decision.optionMass, 1e-6)) });

  return entries
    .sort((left, right) => right.logprob - left.logprob)
    .slice(0, topLogprobs);
}

function generatedAnswer(decision: StubDecision): string {
  const label = decision.labels[decision.winner] ?? "A";
  const mode = decision.hash % 10;

  if (mode === 0) {
    // Exercises the row where a generated answer names no option at all.
    return "It depends on the situation.";
  }

  if (mode === 1) {
    // Exercises the lenient parser.
    return `The answer is ${label}, based on the state.`;
  }

  if (mode === 2) {
    // Exercises disagreement between the two stub paths.
    const other = decision.labels[(decision.winner + 1) % decision.labels.length] ?? label;
    return other;
  }

  return label;
}

function section(prompt: string, start: string, end: string): string {
  const from = prompt.indexOf(start);
  if (from < 0) {
    return "";
  }

  const to = prompt.indexOf(end, from);
  return prompt.slice(from + start.length, to < 0 ? undefined : to);
}

function terms(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((term) => term.length > 3);
}

function softmax(values: number[]): number[] {
  const maximum = Math.max(...values);
  const weights = values.map((value) => Math.exp(value - maximum));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return weights.map((weight) => weight / total);
}

function hashString(text: string): number {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}
