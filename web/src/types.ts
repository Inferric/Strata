export type Run = {
  run_id: string;
  name: string;
  status: string;
  started_at: number;
  model: string;
  kind: string;
  dataset_id?: string;
  split_id?: string;
  site?: string;
  task_kind?: string;
  forecast_horizon_minutes?: string;
  evaluation_partition?: string;
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
    plain_language_question?: string;
    evaluation_partition?: string;
    forecast_horizon_minutes?: number;
    forecast_claim?: {
      eligible: boolean;
      measured_relative_improvement?: number;
      passed_point_estimate?: boolean;
      best_model?: string;
      persistence_rmse_log10_cn2?: number;
      best_neural_rmse_log10_cn2?: number;
    };
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
