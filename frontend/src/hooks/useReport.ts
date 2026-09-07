import { useCallback, useState } from 'react'
import { ApiRequestError, fetchScoutingReport } from '../api/client'
import type { ScoutingReport } from '../types/report'

interface UseReportState {
  report: ScoutingReport | null
  loading: boolean
  error: string | null
}

export function useReport() {
  const [state, setState] = useState<UseReportState>({
    report: null,
    loading: false,
    error: null,
  })

  const search = useCallback(async (playerName: string, season: number) => {
    setState({ report: null, loading: true, error: null })
    try {
      const report = await fetchScoutingReport({ player_name: playerName, season })
      setState({ report, loading: false, error: null })
    } catch (err) {
      const message =
        err instanceof ApiRequestError
          ? err.message
          : 'Could not reach the backend. If this is the first request in a while, ' +
            'the free-tier server may be cold-starting (can take 30-60s) - try again shortly.'
      setState({ report: null, loading: false, error: message })
    }
  }, [])

  return { ...state, search }
}
