import { useEffect, useMemo, useState } from "react";

import { cn } from "./lib/cn";

type Summary = {
  total_clips: number;
  recommended: number;
  needs_review: number;
  low_value: number;
};

type ClipRow = {
  filename: string;
  status: "recommended" | "needs_review" | "low_value";
  training_value_score: number | null;
  quality_flags: string | null;
  curation_reasons: string | null;
  source_title: string | null;
  source_url: string | null;
  demo_category: string | null;
  thumbnail_path: string | null;
  thumbnail_token?: string | null;
  exported_clip_path: string | null;
  hand_visible_ratio: number | null;
  hand_presence_score: number | null;
  hand_motion_score: number | null;
  egocentric_proxy_score: number | null;
  blur_score: number | null;
  brightness_score: number | null;
  camera_stability_score: number | null;
  motion_score: number | null;
  display_w: number | null;
  display_h: number | null;
  duration: number | null;
  effective_fps: number | null;
  source_path: string | null;
};

type RuntimeInfo = {
  defaultInputPath: string;
  detectedClipCount: number;
};

type RunAnalysisResponse = {
  ok: boolean;
  clipCount?: number;
  inputPath?: string;
  summary?: Summary;
  stdout?: string;
  stderr?: string;
  error?: string;
};

type StatusFilter = "all" | "recommended" | "needs_review" | "low_value";
type SortMode = "score_desc" | "score_asc" | "filename_asc";

const STATUS_ORDER: Array<Exclude<StatusFilter, "all">> = [
  "recommended",
  "needs_review",
  "low_value",
];

const STATUS_COPY: Record<Exclude<StatusFilter, "all">, string> = {
  recommended: "RECOMMENDED",
  needs_review: "NEEDS REVIEW",
  low_value: "LOW VALUE",
};

const STATUS_CLASS: Record<Exclude<StatusFilter, "all">, string> = {
  recommended: "border-[var(--accent)] text-[var(--accent)]",
  needs_review: "border-[var(--line-strong)] text-[var(--ink)]",
  low_value: "border-[var(--ink)] text-[var(--paper)] bg-[var(--ink)]",
};

function toAssetPath(pathValue: string | null | undefined): string {
  if (!pathValue) {
    return "";
  }
  if (pathValue.startsWith("/")) {
    return pathValue;
  }
  return `/${pathValue.replace(/^exports\//, "")}`;
}

function withVersion(pathValue: string | null | undefined, token?: string | null): string {
  const base = toAssetPath(pathValue);
  if (!base) {
    return "";
  }
  return token ? `${base}?v=${encodeURIComponent(token)}` : base;
}

function splitTags(value: string | null | undefined): string[] {
  if (!value) {
    return [];
  }
  return value
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatScore(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(3);
}

function formatMetric(value: number | null | undefined, suffix = ""): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "skipped";
  }
  return `${value.toFixed(2)}${suffix}`;
}

async function loadArtifacts(): Promise<[Summary, ClipRow[]]> {
  const [summaryResponse, manifestResponse] = await Promise.all([
    fetch("/summary.json"),
    fetch("/dataset_manifest.json"),
  ]);

  if (!summaryResponse.ok || !manifestResponse.ok) {
    throw new Error("Failed to load dataset artifacts.");
  }

  const [summaryData, manifestData] = await Promise.all([
    summaryResponse.json() as Promise<Summary>,
    manifestResponse.json() as Promise<ClipRow[]>,
  ]);

  return [summaryData, manifestData];
}

function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [clips, setClips] = useState<ClipRow[]>([]);
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimeInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortMode, setSortMode] = useState<SortMode>("score_desc");
  const [inputPath, setInputPath] = useState("");
  const [runState, setRunState] = useState<{
    running: boolean;
    message: string | null;
    stderr: string | null;
  }>({
    running: false,
    message: null,
    stderr: null,
  });
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    let active = true;

    async function bootstrap() {
      setLoading(true);
      setError(null);

      try {
        const [runtimeResponse, artifactData] = await Promise.all([
          fetch("/api/runtime-info"),
          loadArtifacts(),
        ]);

        if (!runtimeResponse.ok) {
          throw new Error("Failed to load runtime configuration.");
        }

        const runtime = (await runtimeResponse.json()) as RuntimeInfo;
        const [summaryData, manifestData] = artifactData;

        if (!active) {
          return;
        }

        setRuntimeInfo(runtime);
        setInputPath(runtime.defaultInputPath);
        setSummary(summaryData);
        setClips(manifestData);
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load the current dataset view.",
        );
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void bootstrap();

    return () => {
      active = false;
    };
  }, []);

  const visibleClips = useMemo(() => {
    const filtered =
      statusFilter === "all"
        ? [...clips]
        : clips.filter((clip) => clip.status === statusFilter);

    filtered.sort((left, right) => {
      if (sortMode === "filename_asc") {
        return left.filename.localeCompare(right.filename);
      }

      const leftScore =
        typeof left.training_value_score === "number"
          ? left.training_value_score
          : -1;
      const rightScore =
        typeof right.training_value_score === "number"
          ? right.training_value_score
          : -1;

      return sortMode === "score_asc"
        ? leftScore - rightScore
        : rightScore - leftScore;
    });

    return filtered;
  }, [clips, sortMode, statusFilter]);

  async function refreshArtifacts() {
    const [summaryData, manifestData] = await loadArtifacts();
    setSummary(summaryData);
    setClips(manifestData);
  }

  async function runAnalysis() {
    setRunState({
      running: true,
      message: "Running pipeline analysis...",
      stderr: null,
    });

    try {
      const response = await fetch("/api/run-analysis", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputPath,
        }),
      });

      const payload = (await response.json()) as RunAnalysisResponse;
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || payload.stderr || "Analysis failed.");
      }

      await refreshArtifacts();
      setRunState({
        running: false,
        message: `Analysis finished for ${payload.clipCount ?? 0} clip(s) from ${payload.inputPath}.`,
        stderr: payload.stderr || null,
      });
    } catch (runError) {
      setRunState({
        running: false,
        message: runError instanceof Error ? runError.message : "Analysis failed.",
        stderr: null,
      });
    }
  }

  async function runUploadedAnalysis() {
    if (selectedFiles.length === 0) {
      setRunState({
        running: false,
        message: "Select one or more local video files first.",
        stderr: null,
      });
      return;
    }

    setRunState({
      running: true,
      message: `Uploading ${selectedFiles.length} file(s) and running pipeline analysis...`,
      stderr: null,
    });

    try {
      const formData = new FormData();
      for (const file of selectedFiles) {
        formData.append("videos", file, file.name);
      }

      const response = await fetch("/api/run-analysis-upload", {
        method: "POST",
        body: formData,
      });

      const payload = (await response.json()) as RunAnalysisResponse;
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || payload.stderr || "Upload analysis failed.");
      }

      await refreshArtifacts();
      setRunState({
        running: false,
        message: `Analysis finished for ${payload.clipCount ?? 0} uploaded clip(s).`,
        stderr: payload.stderr || null,
      });
      setSelectedFiles([]);
    } catch (runError) {
      setRunState({
        running: false,
        message: runError instanceof Error ? runError.message : "Upload analysis failed.",
        stderr: null,
      });
    }
  }

  function addFiles(files: File[]) {
    const videos = files.filter((file) => file.type.startsWith("video/") || /\.[a-z0-9]+$/i.test(file.name));
    if (videos.length === 0) {
      setRunState({
        running: false,
        message: "No supported video files were detected in the dropped selection.",
        stderr: null,
      });
      return;
    }
    setSelectedFiles(videos);
  }

  return (
    <main className="min-h-dvh bg-[var(--paper)] text-[var(--ink)]">
      <div className="mx-auto flex w-full max-w-[1680px] flex-col gap-3 px-3 py-3 sm:px-4">
        <header className="grid gap-px border border-[var(--line-strong)] bg-[var(--line)] lg:grid-cols-[1.4fr_0.9fr]">
          <div className="bg-[var(--panel)] px-4 py-4 sm:px-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">
                  Sieve Egocentric QA Pipeline Demo
                </p>
              </div>
              <div className="shrink-0 border border-[var(--ink)] px-3 py-2 text-right">
                <p className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
                  Current source
                </p>
                <p className="mt-2 max-w-[320px] truncate font-mono text-xs text-[var(--ink)]">
                  {runtimeInfo?.defaultInputPath ?? "loading"}
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-px bg-[var(--line)] sm:grid-cols-2">
            <SummaryCell label="TOTAL" value={summary?.total_clips ?? 0} loading={loading} />
            <SummaryCell label="RECOMMENDED" value={summary?.recommended ?? 0} loading={loading} accent />
            <SummaryCell label="NEEDS REVIEW" value={summary?.needs_review ?? 0} loading={loading} />
            <SummaryCell label="LOW VALUE" value={summary?.low_value ?? 0} loading={loading} inverted />
          </div>
        </header>

        <section className="grid gap-px border border-[var(--line-strong)] bg-[var(--line)] xl:grid-cols-[360px_minmax(0,1fr)]">
          <aside className="bg-[var(--panel)] p-4">
            <div className="space-y-5">
              <div className="space-y-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">
                  Run analysis
                </p>
                <label className="block space-y-2">
                  <span className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                    Folder or single video path
                  </span>
                  <textarea
                    value={inputPath}
                    onChange={(event) => setInputPath(event.target.value)}
                    className="min-h-28 w-full resize-y border border-[var(--line-strong)] bg-[var(--paper)] px-3 py-3 font-mono text-xs leading-6 text-[var(--ink)] outline-none"
                    spellCheck={false}
                  />
                </label>
                <button
                  type="button"
                  onClick={() => void runAnalysis()}
                  disabled={runState.running}
                  className={cn(
                    "w-full border px-3 py-3 text-sm font-medium uppercase tracking-[0.16em]",
                    runState.running
                      ? "border-[var(--line-strong)] bg-[var(--line)] text-[var(--muted)]"
                      : "border-[var(--accent)] bg-[var(--accent)] text-[var(--paper)]",
                  )}
                >
                  {runState.running ? "Running..." : "Run pipeline"}
                </button>
                <p className="text-xs leading-6 text-[var(--muted)]">
                  If you enter a single video path, the runner creates a temporary one-clip input set and reruns the existing Python pipeline against it.
                </p>
                <div className="border-t border-[var(--line)] pt-4">
                  <div className="space-y-2">
                    <span className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">
                      Or drop local video files
                    </span>
                    <label
                      onDragOver={(event) => {
                        event.preventDefault();
                        setDragActive(true);
                      }}
                      onDragLeave={(event) => {
                        event.preventDefault();
                        setDragActive(false);
                      }}
                      onDrop={(event) => {
                        event.preventDefault();
                        setDragActive(false);
                        addFiles(Array.from(event.dataTransfer.files || []));
                      }}
                      className={cn(
                        "block border border-dashed px-3 py-6 text-center text-sm",
                        dragActive
                          ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--ink)]"
                          : "border-[var(--line-strong)] bg-[var(--paper)] text-[var(--muted)]",
                      )}
                    >
                      <input
                        type="file"
                        accept="video/*"
                        multiple
                        onChange={(event) =>
                          addFiles(Array.from(event.target.files || []))
                        }
                        className="hidden"
                      />
                      <span className="block font-medium text-[var(--ink)]">
                        Drag and drop videos here
                      </span>
                      <span className="mt-1 block text-xs uppercase tracking-[0.16em]">
                        or click to choose local files
                      </span>
                    </label>
                  </div>
                  {selectedFiles.length > 0 ? (
                    <div className="mt-2 border border-[var(--line)] bg-[var(--panel)] px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
                        Selected
                      </p>
                      <div className="mt-2 space-y-1">
                        {selectedFiles.map((file) => (
                          <p key={`${file.name}-${file.size}`} className="truncate font-mono text-[11px] text-[var(--ink)]">
                            {file.name}
                          </p>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => void runUploadedAnalysis()}
                    disabled={runState.running || selectedFiles.length === 0}
                    className={cn(
                      "mt-3 w-full border px-3 py-3 text-sm font-medium uppercase tracking-[0.16em]",
                      runState.running || selectedFiles.length === 0
                        ? "border-[var(--line-strong)] bg-[var(--line)] text-[var(--muted)]"
                        : "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]",
                    )}
                  >
                    Run on uploaded files
                  </button>
                </div>
                {runState.message ? (
                  <div className="border border-[var(--line-strong)] bg-[var(--paper)] px-3 py-3 text-xs leading-6 text-[var(--ink)]">
                    {runState.message}
                  </div>
                ) : null}
                {runState.stderr ? (
                  <pre className="max-h-48 overflow-auto border border-[var(--line)] bg-[var(--panel)] px-3 py-3 font-mono text-[10px] leading-5 text-[var(--muted)]">
                    {runState.stderr}
                  </pre>
                ) : null}
              </div>

              <div className="space-y-3 border-t border-[var(--line)] pt-4">
                <p className="text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">
                  Filters
                </p>
                <div className="grid gap-2">
                  <FilterButton
                    active={statusFilter === "all"}
                    label="ALL"
                    count={clips.length}
                    onClick={() => setStatusFilter("all")}
                  />
                  {STATUS_ORDER.map((status) => (
                    <FilterButton
                      key={status}
                      active={statusFilter === status}
                      label={STATUS_COPY[status]}
                      count={clips.filter((clip) => clip.status === status).length}
                      onClick={() => setStatusFilter(status)}
                    />
                  ))}
                </div>
              </div>

              <div className="space-y-3 border-t border-[var(--line)] pt-4">
                <label className="block space-y-2">
                  <span className="text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">
                    Sort
                  </span>
                  <select
                    value={sortMode}
                    onChange={(event) => setSortMode(event.target.value as SortMode)}
                    className="w-full border border-[var(--line-strong)] bg-[var(--paper)] px-3 py-3 text-sm text-[var(--ink)] outline-none"
                  >
                    <option value="score_desc">Training value ↓</option>
                    <option value="score_asc">Training value ↑</option>
                    <option value="filename_asc">Clip ID A → Z</option>
                  </select>
                </label>
              </div>
            </div>
          </aside>

          <section className="bg-[var(--panel)] p-3 sm:p-4">
            {error ? (
              <div className="border border-[var(--ink)] bg-[var(--ink)] px-4 py-3 text-sm text-[var(--paper)]">
                {error}
              </div>
            ) : null}

            {loading ? (
              <div className="grid gap-px border border-[var(--line-strong)] bg-[var(--line)] md:grid-cols-2 2xl:grid-cols-3">
                {Array.from({ length: 6 }).map((_, index) => (
                  <div key={index} className="h-[380px] bg-[var(--paper)]" />
                ))}
              </div>
            ) : (
              <div className="grid gap-px border border-[var(--line-strong)] bg-[var(--line)] md:grid-cols-2 2xl:grid-cols-3">
                {visibleClips.map((clip) => (
                  <ClipPanel key={clip.filename} clip={clip} />
                ))}
              </div>
            )}
          </section>
        </section>
      </div>
    </main>
  );
}

function SummaryCell({
  label,
  value,
  loading,
  accent = false,
  inverted = false,
}: {
  label: string;
  value: number;
  loading: boolean;
  accent?: boolean;
  inverted?: boolean;
}) {
  return (
    <div
      className={cn(
        "px-4 py-4",
        inverted
          ? "bg-[var(--ink)] text-[var(--paper)]"
          : accent
            ? "bg-[var(--accent-soft)] text-[var(--ink)]"
            : "bg-[var(--panel)] text-[var(--ink)]",
      )}
    >
      <p className="text-[10px] uppercase tracking-[0.18em] text-current/70">{label}</p>
      {loading ? (
        <div className="mt-3 h-8 w-16 bg-current/10" />
      ) : (
        <p className="mt-3 font-mono text-4xl tabular-nums">{value}</p>
      )}
    </div>
  );
}

function FilterButton({
  active,
  label,
  count,
  onClick,
}: {
  active: boolean;
  label: string;
  count: number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center justify-between border px-3 py-3 text-left text-sm",
        active
          ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--paper)]"
          : "border-[var(--line-strong)] bg-[var(--paper)] text-[var(--ink)]",
      )}
    >
      <span>{label}</span>
      <span className="font-mono text-xs tabular-nums">{count}</span>
    </button>
  );
}

function ClipPanel({ clip }: { clip: ClipRow }) {
  const qualityFlags = splitTags(clip.quality_flags);
  const curationReasons = splitTags(clip.curation_reasons);

  return (
    <article className="flex flex-col bg-[var(--paper)]">
      <div className="relative aspect-[16/9] border-b border-[var(--line)] bg-black">
        {clip.thumbnail_path ? (
          <img
            src={withVersion(clip.thumbnail_path, clip.thumbnail_token)}
            alt={`${clip.filename} thumbnail`}
            className="h-full w-full object-cover"
          />
        ) : null}
        <div className="absolute left-0 top-0 border-r border-b border-[var(--line-strong)] bg-[var(--paper)] px-3 py-2">
          <p className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
            score
          </p>
          <p className="mt-1 font-mono text-2xl text-[var(--ink)]">
            {formatScore(clip.training_value_score)}
          </p>
        </div>
        <div
          className={cn(
            "absolute right-0 top-0 border-l border-b px-3 py-2 text-[10px] uppercase tracking-[0.16em]",
            STATUS_CLASS[clip.status],
          )}
        >
          {STATUS_COPY[clip.status]}
        </div>
      </div>

      <div className="flex flex-col gap-4 p-4">
        <div className="space-y-1">
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
            {clip.demo_category || "uncategorized"}
          </p>
          <h3 className="text-lg font-medium leading-tight text-[var(--ink)]">
            {clip.source_title || clip.filename}
          </h3>
          <p className="truncate font-mono text-[11px] text-[var(--muted)]">
            {clip.filename}
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-px border border-[var(--line)] bg-[var(--line)]">
          <MetricCell label="resolution" value={`${clip.display_w ?? "?"}×${clip.display_h ?? "?"}`} />
          <MetricCell label="fps" value={formatMetric(clip.effective_fps)} />
          <MetricCell label="duration" value={clip.duration ? `${clip.duration.toFixed(1)}s` : "n/a"} />
          <MetricCell label="brightness" value={formatMetric(clip.brightness_score)} />
          <MetricCell label="hand visible" value={formatMetric(clip.hand_visible_ratio)} />
          <MetricCell label="hand motion" value={formatMetric(clip.hand_motion_score)} />
          <MetricCell label="egocentric" value={formatMetric(clip.egocentric_proxy_score)} />
          <MetricCell label="stability" value={formatMetric(clip.camera_stability_score)} />
        </dl>

        <EvidenceBlock label="Quality Flags" items={qualityFlags} accent={false} />
        <EvidenceBlock label="Curation Reasons" items={curationReasons} accent />

        <div className="flex flex-wrap gap-2 pt-1">
          {clip.source_url ? (
            <a
              href={clip.source_url}
              target="_blank"
              rel="noreferrer"
              className="border border-[var(--line-strong)] px-3 py-2 text-xs uppercase tracking-[0.14em] text-[var(--ink)]"
            >
              Source
            </a>
          ) : null}
          {clip.exported_clip_path ? (
            <a
              href={toAssetPath(clip.exported_clip_path)}
              target="_blank"
              rel="noreferrer"
              className="border border-[var(--accent)] bg-[var(--accent)] px-3 py-2 text-xs uppercase tracking-[0.14em] text-[var(--paper)]"
            >
              Open Clip
            </a>
          ) : null}
        </div>
      </div>
    </article>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[var(--panel)] px-3 py-2">
      <dt className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
        {label}
      </dt>
      <dd className="mt-1 font-mono text-xs text-[var(--ink)] tabular-nums">
        {value}
      </dd>
    </div>
  );
}

function EvidenceBlock({
  label,
  items,
  accent,
}: {
  label: string;
  items: string[];
  accent?: boolean;
}) {
  return (
    <div className="space-y-2">
      <p className="text-[10px] uppercase tracking-[0.18em] text-[var(--muted)]">
        {label}
      </p>
      {items.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {items.map((item) => (
            <span
              key={item}
              className={cn(
                "border px-2 py-1 text-[11px] leading-5",
                accent
                  ? "border-[var(--accent)] text-[var(--accent)]"
                  : "border-[var(--line-strong)] text-[var(--ink)]",
              )}
            >
              {item}
            </span>
          ))}
        </div>
      ) : (
        <p className="text-sm text-[var(--muted)]">none</p>
      )}
    </div>
  );
}

export default App;
