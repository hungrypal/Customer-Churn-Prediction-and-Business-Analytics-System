import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const options = {
  gender: ["Female", "Male"], SeniorCitizen: [0, 1], Partner: ["No", "Yes"], Dependents: ["No", "Yes"],
  PhoneService: ["No", "Yes"], MultipleLines: ["No", "No phone service", "Yes"],
  InternetService: ["DSL", "Fiber optic", "No"],
  OnlineSecurity: ["No", "No internet service", "Yes"], OnlineBackup: ["No", "No internet service", "Yes"],
  DeviceProtection: ["No", "No internet service", "Yes"], TechSupport: ["No", "No internet service", "Yes"],
  StreamingTV: ["No", "No internet service", "Yes"], StreamingMovies: ["No", "No internet service", "Yes"],
  Contract: ["Month-to-month", "One year", "Two year"], PaperlessBilling: ["No", "Yes"],
  PaymentMethod: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
};
const initial = {
  gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 12,
  PhoneService: "Yes", MultipleLines: "No", InternetService: "DSL", OnlineSecurity: "No",
  OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No",
  StreamingMovies: "No", Contract: "Month-to-month", PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check", MonthlyCharges: 79.5, TotalCharges: 954
};
const sections = [
  ["Customer information", ["gender", "SeniorCitizen", "Partner", "Dependents"]],
  ["Services", ["tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]],
  ["Contract & billing", ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]]
];
const label = (name) => name.replace(/([A-Z])/g, " $1").trim().replace("Senior Citizen", "Senior citizen").replace("Streaming T V", "Streaming TV");

export function App() {
  const [values, setValues] = useState(initial);
  const [model, setModel] = useState("random_forest");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const update = (name, value) => setValues((current) => ({ ...current, [name]: value }));

  async function submit(event) {
    event.preventDefault(); setLoading(true); setError(""); setResult(null);
    try {
      const response = await fetch(`${API_URL}/api/v1/predictions/${model}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values)
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail?.[0]?.msg || body.detail || "Prediction failed.");
      setResult(body);
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  }

  return <main className="shell">
    <header><p className="eyebrow">CUSTOMER RETENTION</p><h1>Churn prediction</h1><p>Provide the complete customer profile to score churn risk.</p></header>
    <form onSubmit={submit}>
      {sections.map(([title, fields]) => <section className="form-section" key={title}><h2>{title}</h2><div className="field-grid">
        {fields.map((field) => <label key={field}>{label(field)}
          {options[field] ? <select value={values[field]} onChange={(e) => update(field, field === "SeniorCitizen" ? Number(e.target.value) : e.target.value)}>{options[field].map((value) => <option key={value} value={value}>{field === "SeniorCitizen" ? (value ? "Yes" : "No") : value}</option>)}</select>
            : <input required type="number" min={field === "tenure" ? 0 : field === "MonthlyCharges" ? 18.25 : 18.8} max={field === "tenure" ? 72 : field === "MonthlyCharges" ? 118.75 : 8684.8} step={field === "tenure" ? 1 : 0.01} value={values[field]} onChange={(e) => update(field, Number(e.target.value))} />}
        </label>)}
      </div></section>)}
      <section className="model-section"><h2>Model</h2><div className="model-options">
        {["logistic", "random_forest", "deep_learning"].map((id) => <label key={id}><input type="radio" checked={model === id} onChange={() => setModel(id)} /> {id.replaceAll("_", " ")}</label>)}
      </div><button disabled={loading}>{loading ? "Calculating…" : "Predict churn risk"}</button></section>
    </form>
    {error && <p className="notice error">{error}</p>}
    {result && <section className={`result ${result.churn_risk.toLowerCase()}`}><p>CHURN RISK</p><h2>{result.churn_risk}</h2><strong>{(result.probability * 100).toFixed(1)}%</strong><span>{result.prediction ? "Likely to churn" : "Unlikely to churn"} · {result.model_used.replaceAll("_", " ")}</span></section>}
  </main>;
}
