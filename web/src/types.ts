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
  feature_set?: string;
  horizon_rows?: string;
  horizon_minutes?: string;
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
    plain_language_conclusion?: string;
    assessment_released?: boolean;
    assessment_claim?: {
      eligible: boolean;
      status: string;
      passed?: boolean;
      stronger_operational_control?: string;
      relative_rmse_improvement?: {
        against_history_horizon: number;
        against_stronger_control: number;
        threshold: number;
      };
      conditions?: Record<string, boolean>;
    };
    horizon_matrix?: Array<{
      run_id: string;
      model: string;
      feature_set: string;
      seed: number | null;
      horizon_minutes: number;
      metrics: Record<string, number>;
    }>;
    runs?: Array<{
      run_id: string;
      name: string;
      model: string;
      feature_set: string;
      seed: number | null;
      component_summary?: Record<string, {
        history_delta: { mean: number };
        weather_delta: { mean: number };
        weather_gate: { mean: number };
        weather_contribution: { mean: number };
      }>;
    }>;
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
