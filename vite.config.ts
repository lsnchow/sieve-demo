import Busboy from "busboy";
import {
  createWriteStream,
  mkdtempSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { spawn } from "node:child_process";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import type { Connect, Plugin } from "vite";
import { defineConfig } from "vite";

const repoRoot = "/Users/lucas/Desktop/Sieve2";
const defaultInputPath = "/Users/lucas/Desktop/Sieve M/Sieve/data/raw/diverse_demo";
const videoExtensions = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogv"]);

type Summary = {
  total_clips: number;
  recommended: number;
  needs_review: number;
  low_value: number;
};

function collectVideos(rootPath: string): string[] {
  const stack = [rootPath];
  const files: string[] = [];

  while (stack.length > 0) {
    const current = stack.pop()!;
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (entry.name.startsWith(".")) {
        continue;
      }
      const fullPath = join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (entry.isFile() && videoExtensions.has(extname(entry.name).toLowerCase())) {
        files.push(fullPath);
      }
    }
  }

  return files.sort();
}

function jsonResponse(res: Connect.ServerResponse, statusCode: number, payload: unknown) {
  const body = JSON.stringify(payload);
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(body);
}

function runPipeline(params: {
  inputPath: string;
  clipCount: number;
  onComplete: (result: {
    code: number | null;
    stdout: string;
    stderr: string;
    summary?: Summary;
  }) => void;
}) {
  const tempArtifacts = mkdtempSync(join(tmpdir(), "sieve2-artifacts-"));
  const tempExports = mkdtempSync(join(tmpdir(), "sieve2-exports-"));
  const pythonBin = resolve(repoRoot, ".venv/bin/python");

  const child = spawn(
    pythonBin,
    [
      "run_dataset_demo.py",
      "--artifacts-dir",
      tempArtifacts,
      "run-all",
      "-i",
      params.inputPath,
      "-n",
      String(params.clipCount),
      "--snapshot-output-dir",
      tempExports,
    ],
    {
      cwd: repoRoot,
      env: process.env,
    },
  );

  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += String(chunk);
  });
  child.stderr.on("data", (chunk) => {
    stderr += String(chunk);
  });

  child.on("close", (code) => {
    try {
      if (code === 0) {
        const summary = JSON.parse(
          readFileSync(join(tempExports, "summary.json"), "utf8"),
        ) as Summary;

        rmSync(join(repoRoot, "artifacts"), { recursive: true, force: true });
        rmSync(join(repoRoot, "exports"), { recursive: true, force: true });
        renameSync(tempArtifacts, join(repoRoot, "artifacts"));
        renameSync(tempExports, join(repoRoot, "exports"));

        params.onComplete({
          code,
          stdout,
          stderr,
          summary,
        });
        return;
      }
    } catch (error) {
      stderr += `\n${error instanceof Error ? error.message : String(error)}`;
    } finally {
      rmSync(tempArtifacts, { recursive: true, force: true });
      rmSync(tempExports, { recursive: true, force: true });
    }

    params.onComplete({
      code,
      stdout,
      stderr,
    });
  });
}

function createAnalysisMiddleware(): Connect.NextHandleFunction {
  return (req, res, next) => {
    if (!req.url) {
      return next();
    }

    if (req.method === "GET" && req.url === "/api/runtime-info") {
      const clips = collectVideos(defaultInputPath);
      return jsonResponse(res, 200, {
        defaultInputPath,
        detectedClipCount: clips.length,
      });
    }

    if (req.method === "POST" && req.url === "/api/run-analysis") {
      let body = "";
      req.on("data", (chunk) => {
        body += String(chunk);
      });

      req.on("end", () => {
        try {
          const parsed = JSON.parse(body || "{}") as { inputPath?: string };
          const rawPath = (parsed.inputPath || "").trim();
          const inputPath = rawPath ? resolve(rawPath) : defaultInputPath;
          const extension = extname(inputPath).toLowerCase();
          const isSingleFile = videoExtensions.has(extension);
          const runPath = isSingleFile ? mkdtempSync(join(tmpdir(), "sieve2-single-")) : inputPath;

          if (isSingleFile) {
            symlinkSync(inputPath, join(runPath, "input" + extension));
          }

          const videos = isSingleFile ? [inputPath] : collectVideos(runPath);
          if (videos.length === 0) {
            return jsonResponse(res, 400, {
              ok: false,
              error: "No supported video files found at the requested path.",
            });
          }

          runPipeline({
            inputPath: runPath,
            clipCount: videos.length,
            onComplete: ({ code, stdout, stderr, summary }) => {
            if (isSingleFile) {
              rmSync(runPath, { recursive: true, force: true });
            }

            if (code !== 0) {
              return jsonResponse(res, 500, {
                ok: false,
                code,
                stdout,
                stderr,
              });
            }

            return jsonResponse(res, 200, {
              ok: true,
              inputPath,
              clipCount: videos.length,
              stdout,
              stderr,
              summary,
            });
            },
          });
        } catch (error) {
          return jsonResponse(res, 500, {
            ok: false,
            error: error instanceof Error ? error.message : "Analysis request failed.",
          });
        }
      });

      return;
    }

    if (req.method === "POST" && req.url === "/api/run-analysis-upload") {
      const tempUploadDir = mkdtempSync(join(tmpdir(), "sieve2-upload-"));
      const busboy = Busboy({ headers: req.headers });
      const uploadedFiles: string[] = [];

      busboy.on("file", (_fieldName, file, info) => {
        const safeName = info.filename || `upload${extname(info.filename || ".mp4")}`;
        const targetPath = join(tempUploadDir, safeName);
        uploadedFiles.push(targetPath);
        file.pipe(createWriteStream(targetPath));
      });

      busboy.on("finish", () => {
        if (uploadedFiles.length === 0) {
          rmSync(tempUploadDir, { recursive: true, force: true });
          return jsonResponse(res, 400, {
            ok: false,
            error: "No uploaded video files were received.",
          });
        }

        runPipeline({
          inputPath: tempUploadDir,
          clipCount: uploadedFiles.length,
          onComplete: ({ code, stdout, stderr, summary }) => {
          rmSync(tempUploadDir, { recursive: true, force: true });

          if (code !== 0) {
            return jsonResponse(res, 500, {
              ok: false,
              code,
              stdout,
              stderr,
            });
          }

          return jsonResponse(res, 200, {
            ok: true,
            clipCount: uploadedFiles.length,
            inputPath: "[uploaded files]",
            stdout,
            stderr,
            summary,
          });
          },
        });
      });

      req.pipe(busboy);
      return;
    }

    return next();
  };
}

function localAnalysisPlugin(): Plugin {
  const middleware = createAnalysisMiddleware();

  return {
    name: "local-analysis-runner",
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), localAnalysisPlugin()],
  publicDir: "exports",
});
