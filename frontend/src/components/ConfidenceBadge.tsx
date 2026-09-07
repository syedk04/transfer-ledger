import type { Confidence } from '../types/report'

// Status colors (good/warning/serious), never reused for anything else,
// always paired with a label - never color alone - per the project's
// dataviz color rules.
const STYLES: Record<Confidence, { label: string; bg: string; fg: string }> = {
  high: { label: 'High confidence', bg: '#0ca30c1a', fg: '#0ca30c' },
  medium: { label: 'Medium confidence', bg: '#fab2191a', fg: '#a9740f' },
  low: { label: 'Low confidence', bg: '#d03b3b1a', fg: '#d03b3b' },
}

export function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const style = STYLES[confidence]
  return (
    <span
      className="inline-flex items-center rounded-full px-3 py-1 text-sm font-medium"
      style={{ background: style.bg, color: style.fg }}
    >
      {style.label}
    </span>
  )
}
