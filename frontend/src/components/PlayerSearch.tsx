import { useState } from 'react'

interface Props {
  onSearch: (playerName: string, season: number) => void
  loading: boolean
}

const CURRENT_SEASON = 2025 // matches config.LAST_COMPLETED_SEASON in the backend

export function PlayerSearch({ onSearch, loading }: Props) {
  const [playerName, setPlayerName] = useState('')
  const [season, setSeason] = useState(2024)

  return (
    <form
      className="flex flex-wrap items-end gap-3"
      onSubmit={(e) => {
        e.preventDefault()
        if (playerName.trim()) onSearch(playerName.trim(), season)
      }}
    >
      <label className="flex flex-1 min-w-48 flex-col gap-1">
        <span className="text-sm text-gray-500 dark:text-gray-400">Player</span>
        <input
          className="rounded border border-gray-300 bg-white px-3 py-2 text-black dark:border-gray-600 dark:bg-gray-800 dark:text-white"
          placeholder="e.g. Bukayo Saka"
          value={playerName}
          onChange={(e) => setPlayerName(e.target.value)}
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-sm text-gray-500 dark:text-gray-400">Season</span>
        <input
          type="number"
          className="w-28 rounded border border-gray-300 bg-white px-3 py-2 text-black dark:border-gray-600 dark:bg-gray-800 dark:text-white"
          min={2016}
          max={CURRENT_SEASON}
          value={season}
          onChange={(e) => setSeason(Number(e.target.value))}
        />
      </label>
      <button
        type="submit"
        disabled={loading || !playerName.trim()}
        className="rounded bg-blue-600 px-4 py-2 font-medium text-white disabled:opacity-50"
      >
        {loading ? 'Scouting…' : 'Generate report'}
      </button>
    </form>
  )
}
