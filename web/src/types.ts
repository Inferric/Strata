export type Run = {
  run_id: string;
  name: string;
  status: string;
  started_at: number;
  model: string;
  kind: string;
  dataset_id?: string;
  split_id?: string;
  metrics: Record<string, number>;
  parameters: Record<string, string>;
};

export type Overview = {
  run_count: number;
  completed_count: number;
  best_run: Run | null;
  latest_summary: {
    experiment_id: string;
    hypothesis: string;
    checks: Record<string, boolean>;
    gate_result?: {
      passed: boolean;
      failures: string[];
      best_eligible_model?: string | null;
    };
    report_pdf?: string;
  } | null;
  tracking_source: string;
};

export type SystemStatus = {
  api: string;
  prefect: string;
  mlflow: string;
  hardware_target: string;
  cloud_allowed: boolean;
  cloud_budget_usd: number;
};

export type DatasetManifest = {
  dataset_id: string;
  label_provenance?: string;
  coverage?: Record<string, number | string>;
  terms?: { status: string; redistribution: string };
  verification?: {
    manifest_verified: boolean;
    checks_passed: number;
    checks_total: number;
    split_id?: string;
    split_sha256?: string;
  };
};
