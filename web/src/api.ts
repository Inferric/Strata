import type { DatasetManifest, Overview, Run, SystemStatus } from "./types";

const apiBase = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBase}${path}`);
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}

export async function loadConsole() {
  const [overview, runs, datasets, system] = await Promise.all([
    get<Overview>("/api/overview"),
    get<{ items: Run[] }>("/api/runs"),
    get<{ items: DatasetManifest[] }>("/api/datasets"),
    get<SystemStatus>("/api/system"),
  ]);
  return { overview, runs: runs.items, datasets: datasets.items, system };
}

export const links = {
  mlflow: import.meta.env.VITE_MLFLOW_URL ?? "http://localhost:5000",
  prefect: import.meta.env.VITE_PREFECT_URL ?? "http://localhost:4200",
  api: `${apiBase}/docs`,
  report: `${apiBase}/api/reports/latest`,
};
