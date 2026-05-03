import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { copyFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { createBindMountSandboxProvider } from "@ai-hero/sandcastle";

const IS_WIN32 = process.platform === "win32";

const SH = process.env.SANDCASTLE_SHELL
  || (IS_WIN32 ? "C:\\Program Files\\Git\\bin\\sh.exe" : "sh");

export const hostDirect = (options?: { readonly env?: Record<string, string> }) =>
  createBindMountSandboxProvider({
    name: "host-direct",
    env: options?.env ?? {},
    create: async ({ worktreePath, env }) => {
      const processEnv = { ...process.env, ...env };

      return {
        worktreePath,
        exec: (command, opts) => {
          const cwd = opts?.cwd ?? worktreePath;
          const effectiveCommand = opts?.sudo ? `sudo ${command}` : command;
          return new Promise((resolve, reject) => {
            const proc = spawn(SH, ["-c", effectiveCommand], {
              cwd,
              env: processEnv,
              stdio: [opts?.stdin !== undefined ? "pipe" : "ignore", "pipe", "pipe"],
            });
            if (opts?.stdin !== undefined) { proc.stdin.write(opts.stdin); proc.stdin.end(); }
            const stdoutChunks: string[] = [];
            const stderrChunks: string[] = [];
            if (opts?.onLine) {
              createInterface({ input: proc.stdout }).on("line", (line: string) => {
                stdoutChunks.push(line);
                opts.onLine!(line);
              });
            } else {
              proc.stdout.on("data", (chunk: Buffer) => stdoutChunks.push(chunk.toString()));
            }
            proc.stderr.on("data", (chunk: Buffer) => stderrChunks.push(chunk.toString()));
            proc.on("error", (error: Error) => reject(new Error(`exec failed: ${error.message}`)));
            proc.on("close", (code: number | null) => resolve({
              stdout: stdoutChunks.join(opts?.onLine ? "\n" : ""),
              stderr: stderrChunks.join(""),
              exitCode: code ?? 0,
            }));
          });
        },
        interactiveExec: (args, opts) => new Promise((resolve, reject) => {
          const [cmd, ...rest] = args;
          const proc = spawn(cmd, rest, {
            cwd: opts.cwd ?? worktreePath,
            env: processEnv,
            stdio: [opts.stdin, opts.stdout, opts.stderr],
            shell: IS_WIN32,
          });
          proc.on("error", (error: Error) => reject(new Error(`exec failed: ${error.message}`)));
          proc.on("close", (code: number | null) => resolve({ exitCode: code ?? 0 }));
        }),
        copyFileIn: async (hostPath, sandboxPath) => {
          await mkdir(dirname(sandboxPath), { recursive: true });
          await copyFile(hostPath, sandboxPath);
        },
        copyFileOut: async (sandboxPath, hostPath) => {
          await mkdir(dirname(hostPath), { recursive: true });
          await copyFile(sandboxPath, hostPath);
        },
        close: async () => {},
      };
    },
  });
