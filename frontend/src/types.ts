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
export interface ForecastSeries {
  station: string;
  day: string;
  hours: number[];
  actual: number[];
  predicted: number[];
}
export interface ForecastFile {
  solution_model?: string;
  best_model: string;
  metrics: Metric;
  series: ForecastSeries[];
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
