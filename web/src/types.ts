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

export type GateState = "PASS" | "FAIL" | "NOT_EVALUATED";

export type FusionRunEvidence = {
  run_id: string;
  fold_id: string;
  seed: number;
  primary: Record<string, number>;
  wall_clock_seconds: number;
  peak_total_board_vram_gib: number;
};

export type FusionAggregate = {
  candidate_id: string;
  family: string;
  kind: string;
  parameters?: number | null;
  primary: Record<string, number>;
  fold_primary_rmse: Record<string, number>;
  seed_primary_rmse: Record<string, number>;
  relative_run_rmse_std: number;
  operational?: {
    ood_score_mean: number;
    selective_80_rmse_log10_cn2: number;
    ood_top_quintile_rmse_log10_cn2: number;
    inference_latency_ms_per_sample: number;
    throughput_samples_per_second: number;
    peak_process_reserved_vram_gib?: number;
  };
  diagnostics?: Record<string, {
    residual_mean: number;
    residual_std: number;
    scale_weights_mean: number[];
    expert_weights_mean: number[];
    physics_token_norm_mean: number;
    weather_attention_mean: number;
    modulation_norm_mean: number;
  }>;
  runs: FusionRunEvidence[];
};

export type FusionSummary = {
  experiment_id: string;
  program_id: string;
  task_kind: "fusion_v2_screen" | "fusion_v2_robustness" | "fusion_v2_program";
  status: GateState;
  hypothesis: string;
  plain_language_question: string;
  plain_language_conclusion: string;
  evaluation_partition: string;
  stronger_neural_control?: string;
  aggregates?: Record<string, FusionAggregate>;
  result?: {
    selected_primary_rmse: number;
    persistence_primary_rmse: number;
    stronger_neural_primary_rmse: number;
    diagnostic_lightgbm_primary_rmse?: number;
    relative_improvement_over_persistence: number;
    relative_improvement_over_stronger_neural: number;
    relative_improvement_over_diagnostic_lightgbm?: number;
    seed_directions: Record<string, number>;
    confirmation_eligible: boolean;
    failed_conditions: string[];
  };
  cycles?: Array<{
    cycle_id: string;
    role: string;
    status: GateState;
    question: string;
    conclusion: string;
    report_pdf?: string | null;
  }>;
  ledger_events?: Array<{
    sequence: number;
    recorded_at: string;
    kind: string;
    hypothesis_family: string;
    evidence_role: string;
    result: GateState;
    candidate_id?: string;
    notes: string;
  }>;
  screen_evidence?: {
    run_count: number;
    runs: Array<{
      run_id?: string;
      candidate_id: string;
      category: string;
      family: string;
      primary_rmse: number;
      primary_tail_mae: number;
      primary_coverage_80: number;
    }>;
    closed_hypothesis_families: Record<string, string>;
    selected_for_robustness_characterization: {
      candidate_id?: string;
    };
  };
  checks: Record<string, GateState>;
  resources: {
    matrix_run_count?: number;
    neural_run_count?: number;
    neural_wall_clock_hours: number;
    peak_total_board_vram_gib: number;
    peak_process_allocated_vram_gib?: number;
    artifact_storage_bytes: number;
    cloud_cost_usd: number;
  };
  report_pdf?: string | null;
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
    checks: Record<string, boolean | GateState>;
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
