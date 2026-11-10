# 🚀 Verity Financial RAG - Deployment Guide

This guide provides step-by-step instructions for deploying the **Verity Financial RAG** application to production using **Render** (for the FastAPI backend) and **Vercel** (for the Next.js frontend).

---

## 🏗️ Architecture Summary

- **Backend**: FastAPI + stateless LangGraph (`verity_graph`) running on **Render** (Free Tier Web Service, 512MB RAM).
- **Frontend**: Next.js 14+ App Router + Tailwind CSS running on **Vercel**.
- **Vector Database**: Qdrant Cloud Vector Database (Persistent Index).
- **Hybrid Keyword Search (BM25)**: Fast BM25Okapi index dynamically rebuilt in ~0.5s at startup from committed JSON/NPZ inputs (`backend/artifacts/bm25_inputs.json` / `bm25_inputs.npz`).
- **Embeddings Backend**: Swappable via `EMBEDDING_BACKEND=fastembed` (ONNX Runtime, ~50MB RAM footprint vs ~400MB PyTorch).

---

## 🐍 Part 1: Deploying Backend to Render

### 1. Create Web Service on Render
1. Navigate to the [Render Dashboard](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ **Web Service** (or use the blueprint with `render.yaml`).
3. Connect your GitHub repository containing the Verity codebase.
4. Fill out the service configuration settings:
   - **Name**: `verity-backend`
   - **Region**: Choose closest to your users (e.g., Oregon / Frankfurt).
   - **Branch**: `main` (or your active working branch).
   - **Root Directory**: *(Leave empty / root `.`)* &mdash; **CRITICAL**: Do NOT set to `backend`, so Python module imports (`backend.agent...`) resolve correctly from the repository root.
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `python -m uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free` (512MB RAM)

### 2. Configure Environment Variables
In the Render dashboard for `verity-backend`, navigate to **Environment** $\rightarrow$ **Environment Variables** and add:

| Key | Value Description | Example / Note |
|---|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API Key | `AIzaSy...` |
| `QDRANT_URL` | Qdrant Cloud Cluster Endpoint URL | `https://xxxx.us-west-1-0.aws.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | Qdrant Cluster API Key | `th-xxxx...` |
| `ALLOWED_ORIGIN` | Allowed Frontend CORS Origins | `https://your-app.vercel.app` (or `*` for initial testing) |
| `EMBEDDING_BACKEND` | Embedding engine (`fastembed` or `local`) | `fastembed` *(CRITICAL for Render 512MB RAM limit)* |
| `GEN_MODEL` | Generator/Draft LLM model name | `gemini-3.5-flash-lite` |
| `CRITIC_MODEL` | Critic audit LLM model name | `gemini-3.1-flash-lite` |
| `PYTHONPATH` | Python import path | `.` |

### 3. Deploy & Copy Service URL
1. Click **Deploy Web Service**.
2. Once deployment completes, copy your service URL (e.g., `https://verity-backend.onrender.com`).

> ℹ️ **Note on Free-Tier Cold Starts**: Render spins down free web services after 15 minutes of inactivity. The first request after idle spin-down may take ~45–60 seconds to respond. The frontend automatically displays a friendly *"Waking up backend..."* notice during cold starts.

---

## ⚡ Part 2: Deploying Frontend to Vercel

### 1. Import Repository on Vercel
1. Navigate to the [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Select and import your GitHub repository.

### 2. Configure Project Settings
1. In the **Configure Project** step:
   - **Framework Preset**: Next.js (Auto-detected).
   - **Root Directory**: Click **Edit** and set to `frontend` *(CRITICAL: Must be set to `frontend`)*.

### 3. Set Environment Variables
Expand the **Environment Variables** section and add:

| Key | Value | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://verity-backend.onrender.com` | Your deployed Render backend URL |

### 4. Deploy Application
1. Click **Deploy**.
2. Vercel will build and deploy your Next.js frontend automatically.

---

## 🔄 Part 3: Finalizing CORS Security

Once your Vercel deployment completes and yields a production URL (e.g. `https://verity-rag.vercel.app`):

1. Go back to the **Render Dashboard** $\rightarrow$ `verity-backend` $\rightarrow$ **Environment**.
2. Update `ALLOWED_ORIGIN` to your exact Vercel production domain:
   ```env
   ALLOWED_ORIGIN=https://verity-rag.vercel.app
   ```
3. Save changes (Render will trigger a quick zero-downtime redeploy).

---

## 🧪 Post-Deployment Verification Checklist

Verify the following endpoints immediately after Render deployment:

1. **System Health Check**:
   ```bash
   curl -s https://verity-backend.onrender.com/health
   ```
   - [ ] Must return HTTP 200 with JSON:
     ```json
     {
       "status": "ok",
       "bm25_loaded": true,
       "qdrant_connected": true,
       "chunks": 2921,
       "bm25_matches_index": true
     }
     ```
   - [ ] `bm25_loaded` must be `true` (proves BM25 keyword index rebuilt successfully from `backend/artifacts/bm25_inputs.json`).
   - [ ] `qdrant_connected` must be `true` (proves cloud vector database is reachable).
   - [ ] `chunks` must be `2921`.
   - [ ] `bm25_matches_index` must be `true` (proves BM25 artifact chunk count exactly matches Qdrant vector database points_count).

2. **Corpus Info Check**:
   ```bash
   curl -s https://verity-backend.onrender.com/corpus-info
   ```
   - [ ] Must return HTTP 200 (not 404) with the 5 indexed companies (Apple, Microsoft, Alphabet, Amazon, NVIDIA) and `total_chunks: 2921`.

3. **Evaluation Results Check**:
   ```bash
   curl -s https://verity-backend.onrender.com/eval-results
   ```
   - [ ] Returns baseline and improved benchmark JSON data.

4. **Category Breakdown Check**:
   ```bash
   curl -s https://verity-backend.onrender.com/category-breakdown
   ```
   - [ ] Returns per-category metrics and 95% bootstrap confidence intervals.

5. **Streaming Chat & Citations Check**:
   - [ ] Open the Vercel frontend URL, submit a query (e.g., *"What was Apple's R&D spend in 2024?"*).
   - [ ] Verify streaming status chips (`rewriter` $\rightarrow$ `classifier` $\rightarrow$ `retriever` $\rightarrow$ `draft` $\rightarrow$ `critic` $\rightarrow$ `finalizer`).
   - [ ] Verify hybrid citations appear with company, fiscal year, and chunk IDs.
