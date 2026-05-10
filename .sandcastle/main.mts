// Parallel Planner with Review — four-phase orchestration loop
//
// Phase 1 (Plan): An agent analyzes open issues, builds a
// dependency graph, and outputs a <plan> JSON
// listing unblocked issues with branch names.
// Phase 2 (Execute + Review): For each issue, a sandbox is created via
// createSandbox(). The implementer runs first
// (100 iterations). If it produces commits, a
// reviewer runs in the same sandbox on the same
// branch (1 iteration). All issue pipelines run
// concurrently via Promise.allSettled().
// Phase 3 (Merge): A single agent merges all completed branches
// into the current branch.
//
// The outer loop repeats up to MAX_ITERATIONS times so that newly unblocked
// issues are picked up after each round of merges.
//
// Usage:
// npx tsx .sandcastle/main.mts
// Or add to package.json:
// "scripts": { "sandcastle": "npx tsx .sandcastle/main.mts" }

import * as sandcastle from "@ai-hero/sandcastle";
import { hostDirect } from "./host-direct.mts";

const MAX_ITERATIONS = 10;

const PI_MODEL = "nvidia-nim/z-ai/glm-5.1";

const IDLE_TIMEOUT = 10800;

const sandbox = hostDirect({
  env: {
    PYTHONUNBUFFERED: "1",
    FORCE_COLOR: "0",
    NO_COLOR: "1",
  },
});

const VENV_PIP = ".venv/Scripts/pip";
const hooks = {
  sandbox: {
    onSandboxReady: [
      { command: `${VENV_PIP} install --no-deps -e .`, timeoutMs: 300000 },
    ],
  },
};

const extractTextParts = (content: unknown): string[] => {
  if (!Array.isArray(content)) return [];

  return content.flatMap((part) => {
    if (
      typeof part === "object" &&
      part !== null &&
      (part as { type?: unknown }).type === "text" &&
      typeof (part as { text?: unknown }).text === "string"
    ) {
      return [(part as { text: string }).text];
    }
    return [];
  });
};

const extractAssistantOutput = (stdout: string): string => {
  const assistantTexts: string[] = [];

  for (const line of stdout.split(/\r?\n/)) {
    if (!line.startsWith("{")) continue;

    try {
      const event = JSON.parse(line) as Record<string, unknown>;
      const message =
        typeof event.message === "object" && event.message !== null
          ? (event.message as Record<string, unknown>)
          : undefined;

      if (
        (event.type === "message_start" || event.type === "message_end") &&
        message?.role === "assistant"
      ) {
        assistantTexts.push(...extractTextParts(message.content));
        if (typeof message.errorMessage === "string") {
          assistantTexts.push(message.errorMessage);
        }
        continue;
      }

      if (event.type === "turn_end" && message?.role === "assistant") {
        assistantTexts.push(...extractTextParts(message.content));
        if (typeof message.errorMessage === "string") {
          assistantTexts.push(message.errorMessage);
        }
      }
    } catch {
      // Ignore malformed JSON lines and non-event output.
    }
  }

  return assistantTexts.join("\n");
};

for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
  console.log(`\n=== Iteration ${iteration}/${MAX_ITERATIONS} ===\n`);

  const plan = await sandcastle.run({
    hooks,
    sandbox,
    name: "planner",
    maxIterations: 1,
    idleTimeoutSeconds: IDLE_TIMEOUT,
    agent: sandcastle.pi(PI_MODEL),
    promptFile: "./.sandcastle/plan-prompt.md",
  });

  const plannerOutput = extractAssistantOutput(plan.stdout) || plan.stdout;
  const planMatches = [
    ...plannerOutput.matchAll(/^\s*<plan>\s*\r?\n([\s\S]*?)\r?\n\s*<\/plan>\s*$/gm),
  ];
  if (planMatches.length === 0) {
    throw new Error(
      "Planning agent did not produce a <plan> tag.\n\n" + plannerOutput,
    );
  }
  const planRaw = planMatches[planMatches.length - 1]![1]!;
  const planJson = planRaw.trim().replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/i, "").trim();
  const { issues } = JSON.parse(planJson) as {
    issues: { id: string; title: string; branch: string }[];
  };

  if (issues.length === 0) {
    console.log("No unblocked issues to work on. Exiting.");
    break;
  }

  console.log(
    `Planning complete. ${issues.length} issue(s) to work in parallel:`,
  );
  for (const issue of issues) {
    console.log(` ${issue.id}: ${issue.title} → ${issue.branch}`);
  }

  const settled = await Promise.allSettled(
    issues.map(async (issue) => {
      const sb = await sandcastle.createSandbox({
        branch: issue.branch,
        sandbox,
      });

      try {
        const implement = await sb.run({
          name: "implementer",
          maxIterations: 100,
          idleTimeoutSeconds: IDLE_TIMEOUT,
          agent: sandcastle.codex("gpt-5.4-mini", { effort: "high" }),
          promptFile: "./.sandcastle/implement-prompt.md",
          promptArgs: {
            TASK_ID: issue.id,
            ISSUE_TITLE: issue.title,
            BRANCH: issue.branch,
          },
        });

        if (implement.commits.length > 0) {
          const review = await sb.run({
            name: "reviewer",
            maxIterations: 1,
            idleTimeoutSeconds: IDLE_TIMEOUT,
            agent: sandcastle.codex("gpt-5.4", { effort: "medium" }),
            promptFile: "./.sandcastle/review-prompt.md",
            promptArgs: {
              BRANCH: issue.branch,
            },
          });

          return {
            ...review,
            commits: [...implement.commits, ...review.commits],
          };
        }

        return implement;
      } finally {
        await sb.close();
      }
    }),
  );

  for (const [i, outcome] of settled.entries()) {
    if (outcome.status === "rejected") {
      console.error(
        ` ✗ ${issues[i]!.id} (${issues[i]!.branch}) failed: ${outcome.reason}`,
      );
    }
  }

  const completedIssues = settled
    .map((outcome, i) => ({ outcome, issue: issues[i]! }))
    .filter(
      (entry) =>
        entry.outcome.status === "fulfilled" &&
        entry.outcome.value.commits.length > 0,
    )
    .map((entry) => entry.issue);

  const completedBranches = completedIssues.map((i) => i.branch);

  console.log(
    `\nExecution complete. ${completedBranches.length} branch(es) with commits:`,
  );
  for (const branch of completedBranches) {
    console.log(` ${branch}`);
  }

  if (completedBranches.length === 0) {
    console.log("No commits produced. Nothing to merge.");
    continue;
  }

  await sandcastle.run({
    hooks,
    sandbox,
    name: "merger",
    maxIterations: 1,
    idleTimeoutSeconds: IDLE_TIMEOUT,
    agent: sandcastle.pi(PI_MODEL),
    promptFile: "./.sandcastle/merge-prompt.md",
    promptArgs: {
      BRANCHES: completedBranches.map((b) => `- ${b}`).join("\n"),
      ISSUES: completedIssues
        .map((i) => `- ${i.id}: ${i.title}`)
        .join("\n"),
    },
  });

  console.log("\nBranches merged.");
}

console.log("\nAll done.");
