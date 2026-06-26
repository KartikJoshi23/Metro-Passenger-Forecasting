export default function Footer({ slim = false }: { slim?: boolean }) {
  return (
    <div className={`foot ${slim ? "slim" : ""}`}>
      <span className="credit">
        Built with <span className="heart">❤</span> by <b>Kartik</b> · <b>Prem</b> · <b>Gagandeep</b> · <b>Sam</b>
      </span>
      {!slim && (
        <>
          <br />
          <span className="muted-2">React · TypeScript · Framer Motion · Recharts · TensorFlow/Keras</span><br />
          Data:{" "}
          <a href="https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_ridership-open" target="_blank" rel="noreferrer">
            Dubai Pulse — RTA Metro Ridership (Open Data)
          </a>{" "}· latest 2026 records
        </>
      )}
    </div>
  );
}
