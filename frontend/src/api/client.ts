import type { ApiError, ReportRequest, ScoutingReport } from '../types/report'

// The only backend URL the frontend ever knows - no other secrets/keys live
// client-side, all provider calls (Groq, NewsData) happen server-side.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiRequestError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = status
  }
}

export async function fetchScoutingReport(request: ReportRequest): Promise<ScoutingReport> {
  const response = await fetch(`${API_BASE_URL}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ApiError | null
    throw new ApiRequestError(
      response.status,
      body?.detail ?? `Request failed with status ${response.status}`,
    )
  }

  return (await response.json()) as ScoutingReport
}
