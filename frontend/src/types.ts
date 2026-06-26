export interface Metric {
  MAE: number;
  RMSE: number;
  MAPE: number | null;
  R2: number;
}
export interface MetricsFile {
  metrics: Record<string, Metric>;
  val_mae_curves: Record<string, number[]>;
  solution_model?: string;
  lookback: number;
  test_windows: number;
}
export interface LiveStation {
  station: string;
  line: string;
  total: number;
  actual: number[];
  predicted: number[];
  climatology: number[];
}
export interface LiveFile {
  solution_model: string;
  day: string;
  hours: number[];
  capacity_per_train: number;
  metrics: Metric;
  network: { actual: number[]; predicted: number[] };
  stations: LiveStation[];
}
export interface EdaFile {
  hours: number[];
  profile_all: number[];
  profile_weekday: number[];
  profile_weekend: number[];
  top_stations: { station: string; checkins: number }[];
  line_split: Record<string, number>;
  n_taps: number;
  n_days: number;
  n_stations: number;
}
