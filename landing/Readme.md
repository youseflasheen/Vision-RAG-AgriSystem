# AgriVision AI — Web Frontend

Next.js web interface for the AgriVision AI pest detection system.
Same structure as the compression-algorithms reference project.

## Stack
- **Next.js 14** (App Router)
- **TypeScript**
- **Tailwind CSS**
- **Recharts** — spread simulation charts
- **react-markdown** — renders RAG treatment advice
- **react-dropzone** — image upload

## Local Development

### 1. Install dependencies
```bash
cd landing
npm install
```

### 2. Set the backend URL
Create a `.env.local` file:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Start the FastAPI backend (in a separate terminal)
```bash
# From project root
uvicorn api.main:app --reload --port 8000
```

### 4. Start the Next.js dev server
```bash
npm run dev
```
Open http://localhost:3000

## Deploy to Vercel

1. Push the `landing/` folder to your GitHub repo
2. Go to vercel.com → New Project → import your repo
3. Set **Root Directory** to `landing`
4. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-fastapi-url`
5. Click Deploy

> **Note:** Vercel hosts the frontend only. The FastAPI backend needs to be
> deployed separately (e.g. Railway, Render, or any cloud VM). Update
> `NEXT_PUBLIC_API_URL` in Vercel's environment settings to point to it.

## Project Structure
```
landing/
├── src/
│   └── app/
│       ├── layout.tsx    # Root layout, fonts, metadata
│       ├── page.tsx      # Full app — upload, results, tabs, chatbot
│       └── globals.css   # Tailwind + custom styles
├── package.json
├── tailwind.config.js
├── next.config.js        # API proxy rewrite
├── tsconfig.json
└── vercel.json
```

## API Endpoints Used
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/full_analysis/` | POST | Upload image → full DSS report |
| `/api/chat/` | POST | Chatbot question → answer |
| `/api/` | GET | Health check |

Fix 3 — Next.js Web App ✅ (same structure as compression-algorithms)
Copy the entire landing/ folder into your project root. Then:
bashcd landing
npm install
npm run dev        # runs on localhost:3000
The app has:

Image upload with drag & drop
4 tabs: Detection results, Simulation chart, Treatment advice (markdown), Chatbot
Spread chart (Recharts) showing 3 scenario lines
Intervention ranking table
System overview cards (no "Lab" labels)

To deploy on Vercel (exactly like compression-algorithms-three.vercel.app):

Push your repo to GitHub
Vercel → New Project → set Root Directory to landing
Add env var: NEXT_PUBLIC_API_URL=https://your-backend-url
Deploy — your FastAPI needs to be hosted somewhere separately (Railway or Render are free)