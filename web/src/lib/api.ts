import { SimulationRequest, SimulationResponse } from "./types";

// In production (Vercel), API routes are at the same origin via rewrites.
// In development, hit the FastAPI backend directly.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

export async function runSimulation(
  config: SimulationRequest
): Promise<SimulationResponse> {
  const res = await fetch(`${API_URL}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}
