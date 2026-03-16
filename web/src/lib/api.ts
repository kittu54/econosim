import { SimulationRequest, SimulationResponse, PeriodData } from "./types";

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

// ── Data Persistence ─────────────────────────────────────────────

export interface SavedRun {
  id: string;
  name: string;
  created_at: string;
  config: Record<string, unknown>;
  summary: Record<string, unknown>;
  data?: PeriodData[]; // Explicitly added when fetching detail
}

export async function saveRun(
  name: string,
  config: Record<string, unknown>,
  summary: Record<string, unknown>,
  periods: PeriodData[],
  aggregate: Record<string, unknown>[] | null = null
): Promise<SavedRun> {
  const res = await fetch(`${API_URL}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config, summary, periods, aggregate }),
  });

  if (!res.ok) {
    throw new Error(`Failed to save run: ${res.status}`);
  }

  return res.json();
}

export async function getRuns(): Promise<SavedRun[]> {
  const res = await fetch(`${API_URL}/api/runs`);
  if (!res.ok) {
    throw new Error(`Failed to fetch runs: ${res.status}`);
  }
  return res.json();
}

export async function getRun(id: string): Promise<SavedRun> {
  const res = await fetch(`${API_URL}/api/runs/${id}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch run detail: ${res.status}`);
  }
  return res.json();
}

export async function deleteRun(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/runs/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Failed to delete run: ${res.status}`);
  }
}
