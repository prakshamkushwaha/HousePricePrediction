import { useState } from "react";
import { GlassButton } from "@/components/ui/glass-button";

const FIELDS = [
  "MedInc",
  "HouseAge",
  "AveRooms",
  "AveBedrms",
  "Population",
  "AveOccup",
  "Latitude",
  "Longitude",
];

// Display-only metadata. Keyed by the same internal field names used
// for form state and the API JSON body — those names are never
// changed, only how each field is labeled/hinted in the UI.
const FIELD_META: Record<string, { label: string; placeholder: string }> = {
  MedInc: { label: "Median Income", placeholder: "e.g. 5.0" },
  HouseAge: { label: "House Age", placeholder: "e.g. 25" },
  AveRooms: { label: "Average Rooms", placeholder: "e.g. 6.0" },
  AveBedrms: { label: "Average Bedrooms", placeholder: "e.g. 1.0" },
  Population: { label: "Population", placeholder: "e.g. 1200" },
  AveOccup: { label: "Average Occupancy", placeholder: "e.g. 3.0" },
  Latitude: { label: "Latitude", placeholder: "e.g. 34.05" },
  Longitude: { label: "Longitude", placeholder: "e.g. -118.25" },
};

// Sensible validation rules per field, keyed by the same internal
// field names used everywhere else (never changed).
const FIELD_RULES: Record<string, { isValid: (value: number) => boolean; hint: string }> = {
  MedInc: { isValid: (value) => value > 0, hint: "greater than 0" },
  HouseAge: { isValid: (value) => value >= 0, hint: "0 or greater" },
  AveRooms: { isValid: (value) => value > 0, hint: "greater than 0" },
  AveBedrms: { isValid: (value) => value > 0, hint: "greater than 0" },
  Population: { isValid: (value) => value > 0, hint: "greater than 0" },
  AveOccup: { isValid: (value) => value > 0, hint: "greater than 0" },
  Latitude: { isValid: (value) => value >= -90 && value <= 90, hint: "between -90 and 90" },
  Longitude: { isValid: (value) => value >= -180 && value <= 180, hint: "between -180 and 180" },
};

const KPIS = [
  { label: "Best Model", value: "Random Forest" },
  { label: "R² Score", value: "80.37%" },
  { label: "RMSE", value: "0.507" },
];

const MODEL_PERFORMANCE = [
  { name: "Linear Regression", score: 64.7 },
  { name: "Ridge", score: 64.7 },
  { name: "Random Forest", score: 80.37 },
  { name: "Gradient Boosting", score: 77.68 },
  { name: "XGBoost", score: 77.56 },
];

// Same data, ranked highest R² first for the comparison visualization.
const RANKED_MODEL_PERFORMANCE = [...MODEL_PERFORMANCE].sort(
  (a, b) => b.score - a.score
);

const USD_TO_INR = 88;

const formatInr = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value * USD_TO_INR);

function App() {
  const [formValues, setFormValues] = useState<Record<string, string>>(
    Object.fromEntries(FIELDS.map((field) => [field, ""]))
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);

  const handleFieldChange = (field: string, value: string) => {
    setFormValues((previous) => ({ ...previous, [field]: value }));
  };

  const handleReset = () => {
    setFormValues(Object.fromEntries(FIELDS.map((field) => [field, ""])));
    setPredictedPrice(null);
    setError(null);
  };

  const handlePredict = async () => {
    for (const field of FIELDS) {
      const rawValue = formValues[field].trim();

      if (rawValue === "" || !Number.isFinite(Number(rawValue))) {
        setError("Please enter all property details before predicting.");
        setPredictedPrice(null);
        return;
      }

      const numericValue = Number(rawValue);
      if (!FIELD_RULES[field].isValid(numericValue)) {
        setError(`${FIELD_META[field].label} must be ${FIELD_RULES[field].hint}.`);
        setPredictedPrice(null);
        return;
      }
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          MedInc: Number(formValues.MedInc),
          HouseAge: Number(formValues.HouseAge),
          AveRooms: Number(formValues.AveRooms),
          AveBedrms: Number(formValues.AveBedrms),
          Population: Number(formValues.Population),
          AveOccup: Number(formValues.AveOccup),
          Latitude: Number(formValues.Latitude),
          Longitude: Number(formValues.Longitude),
        }),
      });

      if (!response.ok) {
        throw new Error(`Prediction failed (status ${response.status}).`);
      }

      const data = await response.json();
      setPredictedPrice(data.predicted_price_usd);
    } catch (err) {
      setPredictedPrice(null);
      setError(
        err instanceof Error
          ? err.message
          : "Could not reach the prediction API."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Presentation-only status derived from existing state — does not
  // affect prediction/API logic, just what the status dot/text show.
  const status = isLoading
    ? { text: "Predicting…", dot: "bg-amber-400 animate-pulse" }
    : error
      ? { text: "Needs attention", dot: "bg-red-400" }
      : predictedPrice !== null
        ? { text: "Prediction ready", dot: "bg-emerald-400" }
        : { text: "Ready to predict", dot: "bg-emerald-400" };

  return (
    <div className="flex min-h-svh flex-col items-center gap-10 bg-neutral-950 px-4 py-12 text-neutral-50 sm:px-6 lg:px-10">
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          House Price Prediction
        </h1>
        <p className="mt-2 text-neutral-400">
          Automated property value prediction using machine learning
        </p>
      </div>

      <div className="flex w-full max-w-6xl flex-col gap-8">
        <div className="w-full rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm sm:p-8">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {/* LEFT: Property Details, inputs, Predict button */}
            <div className="flex flex-col gap-5">
              <h2 className="text-lg font-semibold">Property Details</h2>

              <div className="grid grid-cols-2 gap-4">
                {FIELDS.map((field) => (
                  <label key={field} className="flex flex-col gap-1 text-sm">
                    {FIELD_META[field].label}
                    <input
                      type="number"
                      value={formValues[field]}
                      onChange={(event) => handleFieldChange(field, event.target.value)}
                      placeholder={FIELD_META[field].placeholder}
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-neutral-50 outline-none transition-colors hover:border-white/20 hover:bg-white/10 focus:border-neutral-400 focus:bg-white/10"
                    />
                  </label>
                ))}
              </div>

              <div className="mt-2 flex items-center gap-3">
                <GlassButton onClick={handlePredict} disabled={isLoading}>
                  {isLoading ? "Predicting…" : "Predict House Price"}
                </GlassButton>
                <button
                  type="button"
                  onClick={handleReset}
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-neutral-50 outline-none transition-colors hover:border-white/20 hover:bg-white/10 focus:border-neutral-400"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* RIGHT: Estimated Price display */}
            <div className="flex flex-col items-center justify-center gap-3 pt-6 lg:border-l lg:border-white/10 lg:pl-8 lg:pt-0">
              <h2 className="text-lg font-semibold">Estimated Price</h2>

              <div className="flex items-center gap-1.5 text-xs text-neutral-400">
                <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} />
                {status.text}
              </div>

              <span className="text-3xl font-semibold sm:text-4xl">
                {isLoading
                  ? "Predicting…"
                  : predictedPrice !== null
                    ? formatInr(predictedPrice)
                    : "₹ --"}
              </span>

              {error && (
                <p className="text-center text-sm text-red-400">{error}</p>
              )}
            </div>
          </div>
        </div>

        <div className="grid w-full grid-cols-1 gap-6 sm:grid-cols-3">
          {KPIS.map((kpi) => (
            <div
              key={kpi.label}
              className="rounded-2xl border border-white/10 bg-white/5 p-6 text-center backdrop-blur-sm transition-colors hover:border-white/20"
            >
              <p className="text-xs text-neutral-400">{kpi.label}</p>
              <p className="mt-1 text-lg font-semibold">{kpi.value}</p>
            </div>
          ))}
        </div>

        <div className="w-full rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm sm:p-8">
          <div className="mb-6 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h2 className="text-lg font-semibold">Model Comparison</h2>
            <span className="text-xs text-neutral-400">
              R² Score — % of price variance each model explains
            </span>
          </div>

          <div className="grid grid-cols-1 gap-x-10 gap-y-5 lg:grid-cols-2">
            {RANKED_MODEL_PERFORMANCE.map((model) => {
              const isBest = model.name === "Random Forest";
              return (
                <div key={model.name}>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className={isBest ? "font-semibold text-white" : ""}>
                      {isBest ? "⭐ " : ""}
                      {model.name}
                    </span>
                    <span
                      className={
                        isBest
                          ? "font-semibold text-white"
                          : "text-neutral-400"
                      }
                    >
                      {model.score.toFixed(2)}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-white/10">
                    <div
                      className={
                        isBest
                          ? "h-full rounded-full bg-white shadow-[0_0_12px_rgba(255,255,255,0.6)]"
                          : "h-full rounded-full bg-white/60"
                      }
                      style={{ width: `${model.score}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <p className="mt-8 border-t border-white/10 pt-4 text-center text-sm text-neutral-400">
            <span className="font-semibold text-white">Random Forest</span>{" "}
            achieved the highest R² score and is selected as the production
            model.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
