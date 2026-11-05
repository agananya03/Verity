# Verity Financial RAG — Frontend

A Next.js 16 application with React 19 and Tailwind CSS providing an interactive financial intelligence interface for SEC Form 10-K filings.

## Features

- **Real-Time Streaming**: Server-Sent Events (SSE) stream graph execution progress (`status`, `citation`, `answer`, `critic_status`, `done`).
- **Audit Verification State**: Displays draft synthesis with verification badge, showing whether claims are grounded in SEC source filings.
- **Interactive Source Citations**: Links citations directly to SEC EDGAR 10-K filing documents with section and company metadata.
- **Evaluation Dashboard**: Visualizes RAGAS evaluation metrics, faithfulness scores, and category breakdowns.

## Getting Started

1. **Configure Environment Variables**:
   ```bash
   cp .env.local.example .env.local
   ```
   Set `NEXT_PUBLIC_API_URL` to your backend URL (default: `http://localhost:8000`).

2. **Run Development Server**:
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

3. **Build for Production**:
   ```bash
   npm run build
   npm run start
   ```
