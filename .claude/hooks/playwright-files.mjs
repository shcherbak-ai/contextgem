/**
 * A Claude Code hook (PreToolUse, in .claude/settings.json) that keeps the
 * files of the `playwright` browser in one folder. The browser saves a file it
 * names itself in .playwright-mcp/, but takes a name it's given from the
 * project's root: this refuses such a name unless it's in that folder too.
 * See AGENTS.md, "Checking the README in a browser".
 */
import { realpathSync } from "node:fs";
import path from "node:path";
import { text } from "node:stream/consumers";
import { pathToFileURL } from "node:url";

export const FOLDER = ".playwright-mcp";

/**
 * Whether a file is in the folder.
 *
 * @param {string} filename As a tool was given it: taken from the root.
 * @param {string} root The project's root.
 */
export function inFolder(filename, root) {
  const folder = path.resolve(root, FOLDER);
  const fromFolder = path.relative(folder, path.resolve(root, filename));
  return fromFolder !== "" && !fromFolder.startsWith("..") && !path.isAbsolute(fromFolder);
}

/**
 * The hook's answer to a tool call: a refusal that says where the file goes,
 * or nothing, which lets the call through.
 *
 * @param {{tool_input?: {filename?: unknown}, cwd: string}} call What Claude Code sends.
 * @param {string} root The project's root.
 */
export function answer(call, root) {
  const filename = call.tool_input?.filename;
  if (typeof filename !== "string" || inFolder(filename, root)) {
    return null;
  }
  const name = path.basename(filename.replaceAll("\\", "/"));
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: `The browser's files go in ${FOLDER}/: use the filename "${FOLDER}/${name}".`,
    },
  };
}

/**
 * Whether Node was started with this module, as Claude Code starts the hook,
 * rather than a test importing it.
 *
 * @param {string} url The module's own URL: import.meta.url.
 * @param {string | undefined} script What Node was started with: process.argv[1].
 */
export function startedWith(url, script) {
  if (!script) {
    return false;
  }
  // Node gives a module the URL of the file itself, and the script as it was
  // named: through a symlink or a junction to the project, the two differ,
  // and the hook would answer nothing, which lets every call through.
  return url === pathToFileURL(realpathSync(script)).href;
}

if (startedWith(import.meta.url, process.argv[1])) {
  const call = JSON.parse(await text(process.stdin));
  const refusal = answer(call, process.env.CLAUDE_PROJECT_DIR ?? call.cwd);
  if (refusal) {
    console.log(JSON.stringify(refusal));
  }
}
