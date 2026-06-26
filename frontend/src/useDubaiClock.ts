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
}

// Dubai has no DST — a fixed UTC+4 offset, so we can compute it reliably anywhere.
function dubaiDate(): Date {
  const now = new Date();
  const utcMs = now.getTime() + now.getTimezoneOffset() * 60000;
  return new Date(utcMs + 4 * 3600000);
}

const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

export function useDubaiClock(): DubaiClock {
  const [date, setDate] = useState<Date>(dubaiDate());
  useEffect(() => {
    const id = setInterval(() => setDate(dubaiDate()), 1000);
    return () => clearInterval(id);
  }, []);
  const p = (n: number) => String(n).padStart(2, "0");
  const h = date.getHours(), m = date.getMinutes(), s = date.getSeconds(), dow = date.getDay();
  return {
    date, hour: h, minute: m, second: s, dow,
    isWeekend: dow === 6 || dow === 0,
    hhmmss: `${p(h)}:${p(m)}:${p(s)}`,
    hhmm: `${p(h)}:${p(m)}`,
    weekdayName: DAYS[dow],
  };
}
