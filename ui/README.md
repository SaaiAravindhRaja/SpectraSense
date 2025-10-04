SpectraSense UI (Vite + React + Tailwind)

Quick start:

1. Install dependencies (from repo root or inside `ui/`):

```bash
cd ui
npm install
```

2. Run development server:

```bash
npm run dev
```

3. Build static assets for Flask to serve:

```bash
npm run build
```

This will write files to `ui/dist` which `app.py` will serve when present.
UI folder

- `web/` — React or Streamlit demo
- `design/` — assets and style guide

We'll scaffold a responsive web app (React + Tailwind or Streamlit quick demo) depending on deployment targets.