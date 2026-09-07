# Scouting Analyst frontend

React + Vite + TypeScript + Tailwind dashboard for the scouting-report API
in `../backend`. See the repo root README for the full project writeup.

```bash
npm install
cp .env.example .env.local   # point VITE_API_BASE_URL at the running backend
npm run dev
```

`npm run build` type-checks (`tsc -b`) then produces a static `dist/` -
this is what deploys to Cloudflare Pages.
