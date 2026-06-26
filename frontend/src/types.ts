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
export interface ProfileStation {
  station: string;
  line: string;
  total: number;
  typical: number[];   // typical actual demand by hour (climatology)
  forecast: number[];  // model next-hour forecast by hour
}
export interface DayProfile {
  network: { typical: number[]; forecast: number[] };
  stations: ProfileStation[];
}
export interface LiveFile {
  solution_model: string;
  hours: number[];                 // operating hours, e.g. 5..23
  capacity_per_train: number;
  operating: [number, number];     // [openHour, closeHour) e.g. [5, 24]
  metrics: Metric;
  weekday: DayProfile;
  weekend: DayProfile;
  validation: { day: string; corr: number; actual: number[]; predicted: number[] };
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
