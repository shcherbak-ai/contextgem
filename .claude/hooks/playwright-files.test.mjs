/**
 * Tests of the hook, for Node's own runner: pre-commit runs them, or
 * `node --test .claude/hooks/playwright-files.test.mjs`.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmdirSync, symlinkSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, it } from "node:test";
import { pathToFileURL } from "node:url";

import { answer, startedWith } from "./playwright-files.mjs";

const root = "/projects/contextgem";

function call(toolInput) {
  return {
    tool_name: "mcp__playwright__browser_take_screenshot",
    tool_input: toolInput,
    cwd: root,
  };
}

describe("answer", () => {
  it("lets a file in the folder through", () => {
    assert.equal(answer(call({ filename: ".playwright-mcp/readme-390.png" }), root), null);
    assert.equal(answer(call({ filename: ".playwright-mcp/steps/a.js" }), root), null);
  });

  it("lets a call that names no file through", () => {
    assert.equal(answer(call({ url: "http://127.0.0.1:9100/" }), root), null);
  });

  for (const filename of [
    "readme-390.png",
    "screenshots/readme-390.png",
    ".playwright-mcp/../readme-390.png",
    "/tmp/readme-390.png",
    ".playwright-mcp",
  ]) {
    it(`refuses ${filename}, and says where the file goes`, () => {
      const { hookSpecificOutput } = answer(call({ filename }), root);

      assert.equal(hookSpecificOutput.permissionDecision, "deny");
      assert.ok(hookSpecificOutput.permissionDecisionReason.includes('".playwright-mcp/'));
    });
  }

  for (const filename of ["screenshots/readme-390.png", "C:\\Users\\ann\\Desktop\\readme-390.png"]) {
    it(`suggests the name of ${filename} in the folder`, () => {
      const { hookSpecificOutput } = answer(call({ filename }), root);

      assert.ok(hookSpecificOutput.permissionDecisionReason.includes('".playwright-mcp/readme-390.png"'));
    });
  }
});

describe("the hook, started as Claude Code starts it", () => {
  const project = path.resolve(import.meta.dirname, "..", "..");
  const script = path.join(import.meta.dirname, "playwright-files.mjs");
  let folder;

  /** A link to the project: a junction on Windows, which anyone may make. */
  function link() {
    folder = mkdtempSync(path.join(tmpdir(), "playwright-files-"));
    symlinkSync(project, path.join(folder, "project"), "junction");
    return path.join(folder, "project");
  }

  afterEach(() => {
    if (folder) {
      // The link alone, never what it leads to.
      unlinkSync(path.join(folder, "project"));
      rmdirSync(folder);
      folder = undefined;
    }
  });

  /** What the hook prints for a call in the project at this path. */
  function printed(projectPath, toolInput) {
    const hook = spawnSync(process.execPath, [path.join(projectPath, ".claude", "hooks", "playwright-files.mjs")], {
      input: JSON.stringify({ ...call(toolInput), cwd: projectPath }),
      encoding: "utf8",
      env: { ...process.env, CLAUDE_PROJECT_DIR: projectPath },
    });

    assert.equal(hook.stderr, "");
    return hook.stdout;
  }

  it("refuses a file outside the folder, and lets one in it through", () => {
    const refusal = JSON.parse(printed(project, { filename: "readme-390.png" }));

    assert.equal(refusal.hookSpecificOutput.permissionDecision, "deny");
    assert.equal(printed(project, { filename: ".playwright-mcp/readme-390.png" }), "");
  });

  it("refuses it in a project that's reached through a link too", () => {
    // Node then has the hook's file under two names. Answering nothing would
    // let every call through.
    const refusal = JSON.parse(printed(link(), { filename: "readme-390.png" }));

    assert.equal(refusal.hookSpecificOutput.permissionDecision, "deny");
  });

  it("isn't started where Node was given no script, or another", () => {
    const url = pathToFileURL(script).href;

    assert.equal(startedWith(url, undefined), false);
    assert.equal(startedWith(url, import.meta.filename), false);
    assert.equal(startedWith(url, script), true);
  });
});

// A hook that nothing starts passes its own tests and refuses no file.
describe("the project's settings", () => {
  const project = path.resolve(import.meta.dirname, "..", "..");
  const settings = JSON.parse(readFileSync(path.join(project, ".claude", "settings.json"), "utf8"));

  /** The commands that the settings start before a tool of this name. */
  function before(tool) {
    return (settings.hooks?.PreToolUse ?? [])
      .filter((group) => new RegExp(`^(?:${group.matcher})$`).test(tool))
      .flatMap((group) => group.hooks)
      .filter((hook) => hook.type === "command");
  }

  it("start the hook before each of the browser's tools, and before no other tool", () => {
    for (const tool of ["mcp__playwright__browser_take_screenshot", "mcp__playwright__browser_evaluate"]) {
      assert.equal(before(tool).length, 1, tool);
    }
    for (const tool of ["Bash", "Write", "mcp__another__browser_take_screenshot"]) {
      assert.deepEqual(before(tool), [], tool);
    }
  });

  it("start it without a shell, by the project's path, and it then refuses a file outside the folder", () => {
    const [hook] = before("mcp__playwright__browser_take_screenshot");
    // A command with its arguments apart is started as it is, where one line
    // would go through bash on one machine and PowerShell on another.
    assert.equal(hook.command, "node");
    const started = spawnSync(
      process.execPath,
      hook.args.map((arg) => arg.replaceAll("${CLAUDE_PROJECT_DIR}", project)),
      {
        input: JSON.stringify({ ...call({ filename: "readme-390.png" }), cwd: project }),
        encoding: "utf8",
        env: { ...process.env, CLAUDE_PROJECT_DIR: project },
      },
    );

    assert.equal(started.stderr, "");
    assert.equal(JSON.parse(started.stdout).hookSpecificOutput.permissionDecision, "deny");
  });
});
