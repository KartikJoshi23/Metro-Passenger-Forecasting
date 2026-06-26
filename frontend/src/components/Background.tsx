import { motion } from "framer-motion";

/** Animated ambient background: soft glows, grid, and two flowing "metro lines" (Red & Green). */
export default function Background() {
  return (
    <div className="bg-wrap" aria-hidden>
      <div className="bg-glow red" />
      <div className="bg-glow green" />
      <div className="bg-glow blue" />
      <div className="bg-grid" />
      <svg width="100%" height="100%" style={{ position: "absolute", inset: 0 }}>
        {/* faint base lines */}
        <path d="M -50 160 Q 400 60 760 220 T 1500 180" stroke="#e4002b"
          strokeWidth="2" fill="none" opacity="0.18" />
        <path d="M -50 640 Q 500 760 900 600 T 1600 660" stroke="#00a651"
          strokeWidth="2" fill="none" opacity="0.18" />
        {/* flowing dashes travelling along each line */}
        <motion.path d="M -50 160 Q 400 60 760 220 T 1500 180" stroke="#ff3b5c"
          strokeWidth="3" fill="none" strokeLinecap="round" strokeDasharray="6 220"
          initial={{ strokeDashoffset: 0 }} animate={{ strokeDashoffset: -1356 }}
          transition={{ duration: 9, repeat: Infinity, ease: "linear" }} opacity="0.85" />
        <motion.path d="M -50 640 Q 500 760 900 600 T 1600 660" stroke="#2ee08a"
          strokeWidth="3" fill="none" strokeLinecap="round" strokeDasharray="6 260"
          initial={{ strokeDashoffset: 0 }} animate={{ strokeDashoffset: -1456 }}
          transition={{ duration: 12, repeat: Infinity, ease: "linear" }} opacity="0.85" />
      </svg>
    </div>
  );
}
