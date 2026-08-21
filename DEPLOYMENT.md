# 🚀 Verity Financial RAG - Deployment Guide

This guide provides step-by-step instructions for deploying the **Verity Financial RAG** application to production using **Render** (for the FastAPI backend) and **Vercel** (for the Next.js frontend).

---

## 🏗️ Architecture Summary

- **Backend**: FastAPI + LangGraph (`verity_graph`) running on **Render** (Free Tier Web Service).
- **Frontend**: Next.js 14+ App Router + Tailwind CSS running on **Vercel**.
- **Vector Database**: Qdrant Cloud Vector Database (Persistent Index).

---

## 🐍 Part 1: Deploying Backend to Render

### 1. Create Web Service on Render
1. Navigate to the [Render Dashboard](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository containing the Verity codebase.
4. Fill out the service configuration settings:
   - **Name**: `verity-backend`
   - **Region**: Choose closest to your users (e.g., Oregon / Frankfurt).
   - **Branch**: `main` (or your active working branch).
   - **Root Directory**: `backend` *(CRITICAL: Must be set to `backend`)*
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

### 2. Configure Environment Variables
In the Render dashboard for `verity-backend`, navigate to **Environment** $\rightarrow$ **Environment Variables** and add the following keys:

| Key | Value Description | Example / Note |
|---|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API Key | `AIzaSy...` |
| `QDRANT_URL` | Qdrant Cloud Cluster Endpoint URL | `https://xxxx.us-west-1-0.aws.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | Qdrant Cluster API Key | `th-xxxx...` |
| `ALLOWED_ORIGIN` | Allowed Frontend CORS Origins | `https://your-app.vercel.app` (or `*` for initial testing) |

### 3. Deploy & Copy Service URL
1. Click **Deploy Web Service**.
2. Once deployment completes, copy your service URL (e.g., `https://verity-backend.onrender.com`).

> ℹ️ **Note on Free-Tier Cold Starts**: Render spins down free web services after 15 minutes of inactivity. The first request after idle spin-down may take ~45–60 seconds to respond. The frontend `useVerityChat` hook automatically displays a friendly *"Waking up backend..."* notice during cold starts.

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

- [ ] Visit `https://verity-backend.onrender.com/health` $\rightarrow$ should return `{"status":"ok"}`.
- [ ] Visit `https://verity-backend.onrender.com/eval-results` $\rightarrow$ should return benchmark JSON scores.
- [ ] Open the Vercel frontend URL, submit a query (e.g., *"What was Apple's R&D spend in 2024?"*), and verify streaming response & cited SEC sources.
- [ ] Navigate to `/eval` dashboard on frontend and verify Recharts graph rendering.
