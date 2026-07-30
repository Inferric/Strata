import {
  Activity,
  ArrowUpRight,
  Beaker,
  Boxes,
  CircleGauge,
  Cloud,
  Database,
  FileText,
  FlaskConical,
  GitBranch,
  RefreshCw,
  ServerCog,
  ShieldCheck,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { links, loadConsole } from "./api";
import type { DatasetManifest, Overview, Run, SystemStatus } from "./types";

const emptySystem: SystemStatus = {
  api: "connecting",
  prefect: "connecting",
  mlflow: "connecting",
  hardware_target: "RTX 5080 / 16 GB",
  cloud_allowed: false,
  cloud_budget_usd: 0,
};

function statusClass(status: string) {
  return ["healthy", "mlflow", "FINISHED"].includes(status) ? "good" : "muted";
}

function metric(run: Run | null, key: string) {
  if (!run) return "—";
  const value = run.metrics[key]
    ?? run.metrics[`pooled/${key}`]
    ?? run.metrics[`test/${key}`];
  return Number.isFinite(value) ? value.toFixed(4) : "—";
}

function RunTrend({ runs }: { runs: Run[] }) {
  const points = runs
    .map((run) => run.metrics.rmse_log10_cn2 ?? run.metrics["test/rmse_log10_cn2"])
    .filter(Number.isFinite)
    .slice()
    .reverse();
  if (points.length < 2) {
    return (
      <div className="chart-empty">
        <Activity size={20} />
        <span>The measured run trend appears after two completed evaluations.</span>
      </div>
    );
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = Math.max(max - min, 1e-6);
  const polyline = points
    .map((value, index) => {
      const x = 8 + (index / (points.length - 1)) * 344;
      const y = 118 - ((value - min) / span) * 92;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg className="trend" viewBox="0 0 360 132" role="img" aria-label="RMSE by run">
      <defs>
        <linearGradient id="trend-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#42c8b7" stopOpacity=".32" />
          <stop offset="100%" stopColor="#42c8b7" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[24, 70, 116].map((y) => (
        <line key={y} x1="8" x2="352" y1={y} y2={y} className="grid-line" />
      ))}
      <polyline points={polyline} fill="none" className="trend-line" />
      {polyline.split(" ").map((point) => {
        const [cx, cy] = point.split(",");
        return <circle key={point} cx={cx} cy={cy} r="3.5" className="trend-dot" />;
      })}
    </svg>
  );
}

function HorizonMatrix({ summary }: { summary: NonNullable<Overview["latest_summary"]> }) {
  const rows = summary.horizon_matrix ?? [];
  const horizons = [5, 15, 30, 60];
  const identities = Array.from(
    new Set(rows.map((row) => `${row.model}|${row.feature_set}`)),
  );
  const values = rows
    .map((row) => row.metrics.rmse_log10_cn2)
    .filter(Number.isFinite);
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 1;
  const span = Math.max(maximum - minimum, 1e-6);

  return (
    <section id="horizons" className="panel horizon-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">WEATHER TRACE / LOWER IS BETTER</span>
          <h3>RMSE by information source and forecast horizon</h3>
        </div>
        <span className="state-chip">
          {summary.assessment_released ? "assessment" : "selection"}
        </span>
      </div>
      <div className="horizon-grid" role="table" aria-label="Horizon RMSE matrix">
        <div className="horizon-corner" role="columnheader">Model / feature arm</div>
        {horizons.map((horizon) => (
          <div className="horizon-head" role="columnheader" key={horizon}>
            <strong>{horizon}</strong><span>minutes</span>
          </div>
        ))}
        {identities.map((identity) => {
          const [model, featureSet] = identity.split("|");
          return [
            <div className="horizon-label" role="rowheader" key={`${identity}-label`}>
              <strong>{model.replaceAll("_", " ")}</strong>
              <span>{featureSet.replaceAll("_", " ")}</span>
            </div>,
            ...horizons.map((horizon) => {
              const matching = rows.filter(
                (row) => row.model === model
                  && row.feature_set === featureSet
                  && row.horizon_minutes === horizon,
              );
              const mean = matching.length
                ? matching.reduce(
                  (total, row) => total + row.metrics.rmse_log10_cn2,
                  0,
                ) / matching.length
                : undefined;
              const strength = mean === undefined ? 0 : 1 - (mean - minimum) / span;
              return (
                <div
                  className="horizon-cell"
                  role="cell"
                  key={`${identity}-${horizon}`}
                  style={{ "--trace": strength } as CSSProperties}
                >
                  <i />
                  <strong>{mean?.toFixed(4) ?? "—"}</strong>
                  <span>{matching.length > 1 ? `${matching.length} seeds` : "fixed"}</span>
                </div>
              );
            }),
          ];
        })}
      </div>
      <div className="weather-gates">
        <span className="eyebrow">ACTUAL HORIZON V1 WEATHER GATES</span>
        {horizons.map((horizon) => {
          const gates = (summary.runs ?? [])
            .filter(
              (run) => run.model === "strata_ot_horizon"
                && run.feature_set === "operational_weather",
            )
            .map(
              (run) => run.component_summary?.[String(horizon)]?.weather_gate.mean,
            )
            .filter((value): value is number => Number.isFinite(value));
          const mean = gates.length
            ? gates.reduce((total, value) => total + value, 0) / gates.length
            : undefined;
          return (
            <div className="gate-dial" key={horizon}>
              <span>+{horizon}m</span>
              <i><b style={{ width: `${(mean ?? 0) * 100}%` }} /></i>
              <strong>{mean?.toFixed(3) ?? "—"}</strong>
            </div>
          );
        })}
      </div>
      <p className="horizon-conclusion">
        {summary.plain_language_conclusion
          ?? "The conclusion appears only after the frozen matrix is recorded."}
      </p>
    </section>
  );
}

export default function App() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [datasets, setDatasets] = useState<DatasetManifest[]>([]);
  const [system, setSystem] = useState<SystemStatus>(emptySystem);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    try {
      const payload = await loadConsole();
      setOverview(payload.overview);
      setRuns(payload.runs);
      setDatasets(payload.datasets);
      setSystem(payload.system);
      setError("");
      setUpdatedAt(new Date());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Console refresh failed");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 15_000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const best = overview?.best_run ?? null;
  const gates = overview?.latest_summary?.checks ?? {};
  const gateResult = overview?.latest_summary?.gate_result;
  const gatePasses = useMemo(
    () => Object.values(gates).filter(Boolean).length,
    [gates],
  );
  const forecastClaim = overview?.latest_summary?.forecast_claim;
  const measuredChange = forecastClaim?.measured_relative_improvement;
  const assessmentClaim = overview?.latest_summary?.assessment_claim;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><FlaskConical size={19} /></div>
          <div><strong>STRATA—OT</strong><span>RESEARCH SYSTEM</span></div>
        </div>
        <nav>
          <a className="active" href="#overview"><CircleGauge size={17} />Overview</a>
          <a href="#horizons"><Activity size={17} />Horizon matrix</a>
          <a href="#runs"><Beaker size={17} />Experiments</a>
          <a href="#datasets"><Database size={17} />Datasets</a>
          <a href="#evidence"><ShieldCheck size={17} />Evidence gates</a>
          <a href={links.mlflow} target="_blank" rel="noreferrer"><Boxes size={17} />Runs & artifacts</a>
          <a href={links.prefect} target="_blank" rel="noreferrer"><GitBranch size={17} />Flows</a>
        </nav>
        <div className="sidebar-foot">
          <span className="eyebrow">COMPUTE ENVELOPE</span>
          <strong>{system.hardware_target}</strong>
          <span>
            <Cloud size={14} /> Cloud ${system.cloud_budget_usd.toFixed(0)} /{" "}
            {system.cloud_allowed ? "approved" : "disabled"}
          </span>
        </div>
      </aside>

      <main>
        <header>
          <div>
            <span className="eyebrow">LIVE RESEARCH CONTROL PLANE</span>
            <h1>Optical turbulence, made inspectable.</h1>
          </div>
          <div className="header-actions">
            <span className={`service-pill ${statusClass(system.api)}`}>
              <i /> API {system.api}
            </span>
            <button onClick={() => void refresh()} aria-label="Refresh console">
              <RefreshCw size={16} /> Refresh
            </button>
          </div>
        </header>

        {error && <div className="error-banner">{error}. The console never substitutes demo runs.</div>}

        <section id="overview" className="hero-grid">
          <article className="hero-card">
            <div className="card-top">
              <div>
                <span className="eyebrow">CURRENT RESEARCH STATE</span>
                <h2>{overview?.latest_summary?.experiment_id ?? "Awaiting first real run"}</h2>
              </div>
              <span className="state-chip">{overview?.completed_count ?? 0} complete</span>
            </div>
            <p>
              {overview?.latest_summary?.hypothesis ??
                "Run the guarded Mauna Loa experiment to populate this console with measured evidence."}
            </p>
            <div className="hero-metrics">
              <div><span>Best blocked RMSE</span><strong>{metric(best, "rmse_log10_cn2")}</strong></div>
              <div><span>Best model</span><strong>{best?.model ?? "—"}</strong></div>
              <div>
                <strong>
                  {gateResult?.passed
                    ? "PASS"
                    : gateResult
                      ? `BLOCKED · ${gateResult.failures.length}`
                      : `${gatePasses}/${Object.keys(gates).length || "—"}`}
                </strong>
              </div>
            </div>
          </article>

          <article className="panel trend-card">
            <div className="panel-heading">
              <div><span className="eyebrow">MEASURED ONLY</span><h3>Blocked RMSE trend</h3></div>
              <Activity size={18} />
            </div>
            <RunTrend runs={runs} />
          </article>
        </section>

        {overview?.latest_summary?.plain_language_question && (
          <section className="translation-strip" aria-label="Experiment in plain language">
            <article>
              <span className="translation-number">01</span>
              <div>
                <span className="eyebrow">THE QUESTION</span>
                <p>{overview.latest_summary.plain_language_question}</p>
              </div>
            </article>
            <article>
              <span className="translation-number">02</span>
              <div>
                <span className="eyebrow">THE FAIR COMPARISON</span>
                <p>
                  Persistence copies the latest measurement. Every neural model sees
                  the same six-step history, so beating it would mean learning useful
                  change—not receiving extra information.
                </p>
              </div>
            </article>
            <article>
              <span className="translation-number">03</span>
              <div>
                <span className="eyebrow">THE CURRENT VERDICT</span>
                <p>
                  {assessmentClaim?.eligible
                    ? `${assessmentClaim.status.replaceAll("_", " ")} evidence across 15, 30, and 60 minutes.`
                    : forecastClaim?.eligible && measuredChange !== undefined
                    ? `${forecastClaim.best_model} changed error by ${(measuredChange * 100).toFixed(1)}% versus persistence on the ${overview.latest_summary.evaluation_partition} block.`
                    : "The verdict appears only after comparable measured runs finish."}
                </p>
              </div>
            </article>
          </section>
        )}

        {overview?.latest_summary?.horizon_matrix?.length ? (
          <HorizonMatrix summary={overview.latest_summary} />
        ) : null}

        <section className="service-row">
          {[
            ["MLflow", system.mlflow, links.mlflow],
            ["Prefect", system.prefect, links.prefect],
            ["Research API", system.api, links.api],
          ].map(([name, status, href]) => (
            <a className="service-card" href={href} target="_blank" rel="noreferrer" key={name}>
              <div><ServerCog size={18} /><span><strong>{name}</strong><small>{status}</small></span></div>
              <ArrowUpRight size={16} />
            </a>
          ))}
          {overview?.latest_summary?.report_pdf && (
            <a className="service-card" href={links.report} target="_blank" rel="noreferrer">
              <div>
                <FileText size={18} />
                <span><strong>Latest report</strong><small>LaTeX / PDF</small></span>
              </div>
              <ArrowUpRight size={16} />
            </a>
          )}
        </section>

        <section id="runs" className="panel runs-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">EXPERIMENT LEDGER</span><h3>Recent runs</h3></div>
            <span className="soft">{runs.length} recorded</span>
          </div>
          {runs.length ? (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Run</th><th>Model / task</th><th>Status</th><th>RMSE</th><th>MAE</th><th>VRAM</th></tr></thead>
                <tbody>
                  {runs.slice(0, 10).map((run) => (
                    <tr key={run.run_id}>
                      <td><strong>{run.name}</strong><small>{run.run_id.slice(0, 10)}</small></td>
                      <td>
                        <span className="model-tag">{run.model}</span>
                        <small>
                          {run.site ?? run.dataset_id ?? "recorded dataset"}
                          {run.forecast_horizon_minutes
                            ? ` · +${run.forecast_horizon_minutes} min · ${run.evaluation_partition}`
                            : ""}
                        </small>
                      </td>
                      <td><span className={`run-status ${statusClass(run.status)}`}><i />{run.status}</span></td>
                      <td>{metric(run, "rmse_log10_cn2")}</td>
                      <td>{metric(run, "mae_log10_cn2")}</td>
                      <td>{metric(run, "peak_vram_gb")} GB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <Beaker size={25} />
              <div><strong>No run has been recorded yet.</strong><span>Execute <code>./scripts/first-real-run.sh</code>; synthetic placeholders are disabled.</span></div>
            </div>
          )}
        </section>

        <section className="lower-grid">
          <article id="evidence" className="panel">
            <div className="panel-heading">
              <div><span className="eyebrow">SEALED CRITERIA</span><h3>Evidence gates</h3></div>
              <ShieldCheck size={18} />
            </div>
            <div className="gate-list">
              {Object.keys(gates).length ? Object.entries(gates).map(([name, passed]) => (
                <div key={name}><i className={passed ? "pass" : "fail"} /><span>{name.replaceAll("_", " ")}</span><strong>{passed ? "pass" : "open"}</strong></div>
              )) : <p className="soft">Gates appear after the first report is compiled.</p>}
              {gateResult?.failures.map((failure) => (
                <div key={failure}>
                  <i className="fail" />
                  <span>{failure.replaceAll("_", " ")}</span>
                  <strong>blocked</strong>
                </div>
              ))}
            </div>
          </article>

          <article id="datasets" className="panel">
            <div className="panel-heading">
              <div><span className="eyebrow">PROVENANCE REGISTRY</span><h3>Datasets</h3></div>
              <Database size={18} />
            </div>
            <div className="dataset-list">
              {datasets.length ? datasets.map((dataset) => (
                <div key={dataset.dataset_id}>
                  <div>
                    <strong>{dataset.dataset_id}</strong>
                    <span>
                      {dataset.label_provenance?.replaceAll("_", " ")}
                      {" · "}
                      terms {dataset.terms?.status?.replaceAll("_", " ") ?? "unknown"}
                    </span>
                  </div>
                  <span className="state-chip">
                    {dataset.verification?.manifest_verified
                      ? `verified ${dataset.verification.checks_passed}/${dataset.verification.checks_total}`
                      : "unverified"}
                  </span>
                </div>
              )) : <p className="soft">No manifest acquired yet.</p>}
            </div>
          </article>

          <article className="panel next-panel">
            <div className="panel-heading">
              <div><span className="eyebrow">AUTONOMY BOUNDARY</span><h3>Next iteration</h3></div>
              <GitBranch size={18} />
            </div>
            <div className="next-step">
              <span>1</span><p><strong>Codex proposes</strong>Schema-constrained hypothesis and bounded change.</p>
            </div>
            <div className="next-step">
              <span>2</span><p><strong>Controller verifies</strong>Frozen split, cost, leakage, and parameter allowlist.</p>
            </div>
            <div className="next-step">
              <span>3</span><p><strong>Human promotes</strong>No candidate becomes champion automatically.</p>
            </div>
          </article>
        </section>

        <footer>
          <span><FileText size={14} /> LaTeX-first evidence trail</span>
          <span>{updatedAt ? `Updated ${updatedAt.toLocaleTimeString()}` : "Connecting…"}</span>
        </footer>
      </main>
    </div>
  );
}
