import type { KeyFactor } from '../types/report'

// Diverging bar chart: SHAP contributions push a valuation up (blue) or down
// (red) relative to a zero baseline. Palette is the validated diverging pair
// from the project's dataviz reference (blue<->red, neutral gray midpoint) -
// not picked freehand. Values are shown as log-scale SHAP magnitudes, not EUR
// (see backend/src/explain.py's docstring for why a EUR-scale swing isn't a
// real counterfactual here) - bars communicate relative rank and direction,
// the number is a qualitative "how much bigger than the others", not a
// dollar amount.
export function ShapChart({ factors }: { factors: KeyFactor[] }) {
  if (factors.length === 0) {
    return <p className="text-sm text-[var(--muted)]">No SHAP factors available for this report.</p>
  }

  return (
    <div className="shap-chart flex flex-col gap-2">
      <style>{`
        .shap-chart {
          --surface: #fcfcfb;
          --text-primary: #0b0b0b;
          --text-secondary: #52514e;
          --pos: #2a78d6;
          --neg: #e34948;
          --baseline: #c3c2b7;
        }
        @media (prefers-color-scheme: dark) {
          :root:not([data-theme='light']) .shap-chart {
            --surface: #1a1a19;
            --text-primary: #ffffff;
            --text-secondary: #c3c2b7;
            --pos: #3987e5;
            --neg: #e66767;
            --baseline: #383835;
          }
        }
        :root[data-theme='dark'] .shap-chart {
          --surface: #1a1a19;
          --text-primary: #ffffff;
          --text-secondary: #c3c2b7;
          --pos: #3987e5;
          --neg: #e66767;
          --baseline: #383835;
        }
      `}</style>
      {factors.map((factor) => (
        <ShapBar key={factor.feature} factor={factor} maxAbs={maxAbsShap(factors)} />
      ))}
    </div>
  )
}

function maxAbsShap(factors: KeyFactor[]): number {
  // magnitude_rank orders factors but doesn't carry the raw value into the
  // frontend - bar width is therefore relative rank-based (widest = rank 1),
  // not a literal proportional SHAP magnitude. Good enough for "which
  // factors mattered most, and in which direction" without over-claiming
  // false precision on a log-scale number a reader can't sanity-check anyway.
  return factors.length
}

function ShapBar({ factor, maxAbs }: { factor: KeyFactor; maxAbs: number }) {
  // Bars are anchored at the 50% center line and extend outward toward
  // ONE edge only, so the usable range is 0-50% of the track's full
  // width, not 0-100% - halving here is what keeps the widest (rank 1)
  // bar from overflowing past the track into the sign indicator next to
  // it (a real bug caught by actually rendering this and looking at it,
  // not just reading the code).
  const halfWidthPct = Math.max(6, 50 * (1 - (factor.magnitude_rank - 1) / maxAbs))
  const isPositive = factor.direction === 'positive'

  return (
    <div className="flex items-center gap-3" title={factor.explanation}>
      <span className="w-36 shrink-0 truncate text-right text-sm text-[var(--text-secondary)]">
        {factor.feature}
      </span>
      <div className="relative h-5 flex-1 overflow-hidden rounded bg-[var(--baseline)]/20">
        <div
          className="absolute top-0 h-5 rounded"
          style={{
            width: `${halfWidthPct}%`,
            left: isPositive ? '50%' : undefined,
            right: isPositive ? undefined : '50%',
            background: isPositive ? 'var(--pos)' : 'var(--neg)',
            borderRadius: 4,
          }}
        />
        <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--baseline)]" />
      </div>
      <span
        className="w-6 shrink-0 text-sm font-semibold"
        style={{ color: isPositive ? 'var(--pos)' : 'var(--neg)' }}
      >
        {isPositive ? '+' : '−'}
      </span>
    </div>
  )
}
