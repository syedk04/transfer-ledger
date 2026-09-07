import type { ScoutingReport } from '../types/report'
import { ConfidenceBadge } from './ConfidenceBadge'
import { ShapChart } from './ShapChart'
import { ExportButton } from './ExportButton'

function formatEur(value: number): string {
  return new Intl.NumberFormat('en-GB', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 0,
  }).format(value)
}

export function ScoutingReportView({ report }: { report: ScoutingReport }) {
  return (
    <article id="scouting-report" className="flex flex-col gap-6 rounded-lg border border-gray-200 p-6 dark:border-gray-700">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold">{report.player}</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {report.season}-{(report.season + 1) % 100} season · model: {report.model_used}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ConfidenceBadge confidence={report.confidence} />
          <ExportButton />
        </div>
      </header>

      <div>
        <p className="text-sm text-gray-500 dark:text-gray-400">Predicted market value</p>
        <p className="text-4xl font-bold">{formatEur(report.predicted_value_eur)}</p>
      </div>

      <section>
        <h3 className="mb-1 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
          Why: confidence reasoning
        </h3>
        <p>{report.confidence_reasoning}</p>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
          Top SHAP factors
        </h3>
        <ShapChart factors={report.key_factors} />
        <ul className="mt-3 flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-300">
          {report.key_factors.map((factor) => (
            <li key={factor.feature}>
              <span className="font-medium">{factor.feature}</span>: {factor.explanation}
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
          Recent news context
        </h3>
        {report.news_context.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No recent news articles were found or cited for this report.
          </p>
        ) : (
          <ol className="flex flex-col gap-2 text-sm">
            {report.news_context.map((citation, i) => (
              <li key={citation.url} id={`citation-${i + 1}`}>
                <span className="font-medium">[{i + 1}]</span>{' '}
                <a
                  href={citation.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-blue-600 underline dark:text-blue-400"
                >
                  {citation.title}
                </a>{' '}
                <span className="text-gray-500 dark:text-gray-400">— {citation.source}</span>
                <p className="text-gray-600 dark:text-gray-300">{citation.snippet}</p>
              </li>
            ))}
          </ol>
        )}
      </section>

      {report.caveats.length > 0 && (
        <section>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
            Caveats
          </h3>
          <ul className="list-disc pl-5 text-sm text-gray-600 dark:text-gray-300">
            {report.caveats.map((caveat, i) => (
              <li key={i}>{caveat}</li>
            ))}
          </ul>
        </section>
      )}

      <p className="text-xs text-gray-400">
        Generated {new Date(report.generated_at).toLocaleString()}
      </p>
    </article>
  )
}
