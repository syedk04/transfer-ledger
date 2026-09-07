// Hand-kept in sync with backend/schemas.py's ScoutingReport - there is no
// codegen step (e.g. openapi-typescript) in this project, so a change to
// the Pydantic schema needs a matching manual edit here. Flagged in the
// README as a known drift risk rather than hidden.

export type Confidence = 'low' | 'medium' | 'high'

export interface KeyFactor {
  feature: string
  direction: 'positive' | 'negative'
  magnitude_rank: number
  explanation: string
}

export interface NewsCitation {
  title: string
  source: string
  url: string
  snippet: string
  published_at: string | null
}

export interface ScoutingReport {
  player: string
  season: number
  predicted_value_eur: number
  model_used: string
  confidence: Confidence
  confidence_reasoning: string
  key_factors: KeyFactor[]
  news_context: NewsCitation[]
  caveats: string[]
  generated_at: string
}

export interface ReportRequest {
  player_name: string
  season: number
}

export interface ApiError {
  detail: string
}
