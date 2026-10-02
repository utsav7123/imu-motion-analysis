"use strict";

const number = new Intl.NumberFormat("en-CA");
const percent = new Intl.NumberFormat("en-CA", { style: "percent", maximumFractionDigits: 1 });

async function loadSummary() {
  const status = document.getElementById("status");
  try {
    const response = await fetch("./data/summary.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Request failed with ${response.status}`);
    const data = await response.json();
    document.getElementById("rows").textContent = number.format(data.rows);
    document.getElementById("sampling").textContent = `${data.sampling_hz} Hz`;
    document.getElementById("windows").textContent = number.format(data.window_count);
    document.getElementById("f1").textContent = percent.format(data.macro_f1);
    document.getElementById("model").textContent = data.best_model;
    status.textContent = `Analysis loaded from ${data.source_file}.`;
  } catch (error) {
    status.textContent = "The analysis summary is not available yet. Run the pipeline to generate it.";
  }
}

loadSummary();
