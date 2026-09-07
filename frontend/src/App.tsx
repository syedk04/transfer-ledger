import { PlayerSearch } from './components/PlayerSearch'
import { ScoutingReportView } from './components/ScoutingReportView'
import { useReport } from './hooks/useReport'

function App() {
  const { report, loading, error, search } = useReport()

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-6 text-gray-900 dark:text-gray-100">
      <header>
        <h1 className="text-3xl font-bold">Scouting Analyst</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Premier League transfer-value predictions, grounded in SHAP feature attributions and
          cited recent news - not invented.
        </p>
      </header>

      <PlayerSearch onSearch={search} loading={loading} />

      {loading && (
        <p className="text-gray-500 dark:text-gray-400">
          Running the valuation, SHAP explanation, and agent synthesis - if the backend has been
          idle, the free-tier server may need ~30-60s to cold-start first.
        </p>
      )}

      {error && (
        <p className="rounded border border-red-300 bg-red-50 p-3 text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      {report && <ScoutingReportView report={report} />}
    </div>
  )
}

export default App
