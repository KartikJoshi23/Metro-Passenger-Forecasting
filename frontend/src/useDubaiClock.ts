import { useState, useEffect } from "react";

export interface DubaiClock {
  date: Date;        // current time in Dubai (UTC+4)
  hour: number;      // 0..23
  minute: number;
  second: number;
  dow: number;       // 0=Sun .. 6=Sat
  isWeekend: boolean;// UAE weekend = Sat/Sun
  hhmmss: string;
  hhmm: string;
  weekdayName: string;
  isDemo: boolean;   // true when a presentation time-override is active
}

/** Optional presentation override. When null fields are passed, the real value is used.
 *  Leaving the override undefined entirely keeps the clock 100% real (default behaviour). */
export interface ClockOverride {
  hour: number | null;
  weekend: boolean | null;
}

// Dubai has no DST — a fixed UTC+4 offset, so we can compute it reliably anywhere.
function dubaiDate(): Date {
  const now = new Date();
  const utcMs = now.getTime() + now.getTimezoneOffset() * 60000;
  return new Date(utcMs + 4 * 3600000);
}

const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

export function useDubaiClock(override?: ClockOverride): DubaiClock {
  const [date, setDate] = useState<Date>(dubaiDate());
  useEffect(() => {
    const id = setInterval(() => setDate(dubaiDate()), 1000);
    return () => clearInterval(id);
  }, []);
  const p = (n: number) => String(n).padStart(2, "0");

  // real values
  let h = date.getHours();
  const m = date.getMinutes(), s = date.getSeconds();
  let dow = date.getDay();
  let isDemo = false;

  // apply optional presentation override (minutes/seconds keep ticking for a live feel)
  if (override) {
    if (override.hour != null) { h = override.hour; isDemo = true; }
    if (override.weekend != null) { dow = override.weekend ? 6 : 2; isDemo = true; } // Sat / Tue
  }

  return {
    date, hour: h, minute: m, second: s, dow,
    isWeekend: dow === 6 || dow === 0,
    hhmmss: `${p(h)}:${p(m)}:${p(s)}`,
    hhmm: `${p(h)}:${p(m)}`,
    weekdayName: DAYS[dow],
    isDemo,
  };
}
