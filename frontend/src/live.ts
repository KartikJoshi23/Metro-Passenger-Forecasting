import type { LiveFile, DayProfile } from "./types";
import type { DubaiClock } from "./useDubaiClock";

export interface LiveState {
  open: boolean;
  profile: DayProfile;
  hours: number[];
  nowHour: number;
  nowIdx: number;        // index into hours for the current hour (-1 if closed)
  nextHour: number;
  closing: boolean;      // current hour is the last operating hour
  openHour: number;
  closeHour: number;
}

export function liveState(live: LiveFile, clock: DubaiClock): LiveState {
  const [openHour, closeHour] = live.operating;
  const profile = clock.isWeekend ? live.weekend : live.weekday;
  const open = clock.hour >= openHour && clock.hour < closeHour;
  const nowIdx = open ? live.hours.indexOf(clock.hour) : -1;
  const closing = clock.hour === closeHour - 1;
  return {
    open, profile, hours: live.hours, nowHour: clock.hour, nowIdx,
    nextHour: (clock.hour + 1) % 24, closing, openHour, closeHour,
  };
}

// demand band relative to the day's peak (for qualitative crowding labels)
export function band(value: number, peak: number): { label: string; color: string } {
  const r = peak > 0 ? value / peak : 0;
  if (r >= 0.85) return { label: "Peak", color: "#ff3b5c" };
  if (r >= 0.6) return { label: "High", color: "#f4b740" };
  if (r >= 0.35) return { label: "Moderate", color: "#9cc0ff" };
  return { label: "Low", color: "#2ee08a" };
}

export const fmt = (n: number) => Math.round(n).toLocaleString();
