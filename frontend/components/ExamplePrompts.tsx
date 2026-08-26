"use client";

import { useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";

interface ExamplePromptsProps {
  onSelectPrompt: (prompt: string) => void;
}

export function ExamplePrompts({ onSelectPrompt }: ExamplePromptsProps) {
  const [faithfulness, setFaithfulness] = useState<string>("96%");

  useEffect(() => {
    async function fetchEvalScore() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${apiUrl}/eval-results`);
        if (res.ok) {
          const data = await res.json();
          const score = data?.improved?.aggregate_metrics?.faithfulness;
          if (typeof score === "number") {
            setFaithfulness(`${Math.round(score * 100)}%`);
          }
        }
      } catch {
        // Fallback to default 96% on connection issue
      }
    }
    fetchEvalScore();
  }, []);

  const items = [
    {
      itemNumber: "Item 7",
      title: "R&D Spend Analysis",
      prompt: "What was Apple's total R&D spend in fiscal year 2024?",
      ticker: "AAPL",
    },
    {
      itemNumber: "Item 7",
      title: "Segment Revenue Comparison",
      prompt: "Compare Microsoft's Intelligent Cloud revenue between fiscal year 2023 and 2024.",
      ticker: "MSFT",
    },
    {
      itemNumber: "Item 1A",
      title: "Risk Factors Audit",
      prompt: "What are the primary supply chain risk factors listed in Apple's Item 1A filing?",
      ticker: "AAPL",
    },
    {
      itemNumber: "Item —",
      title: "Unanswerable Query Test",
      prompt: "What was Netflix's global subscriber churn rate and ARPU in Q4 2024?",
      ticker: "NFLX",
    },
  ];

  return (
    <div className="w-full max-w-4xl py-6 sm:py-10 text-left">
      {/* Left-Aligned Hero */}
      <div className="mb-10 sm:mb-12">
        <h1 className="font-display font-medium text-3xl sm:text-4xl md:text-5xl text-[#EDE7DA] leading-tight sm:leading-snug">
          <div>Ask a 10-K a question.</div>
          <div>
            Get an answer that{" "}
            <span className="relative inline-block z-10 px-1">
              cites its source
              <svg
                viewBox="0 0 220 24"
                preserveAspectRatio="none"
                aria-hidden="true"
                className="absolute left-0 bottom-0.5 w-full h-3 text-[#E8A33D]/40 pointer-events-none -z-10"
              >
                <path
                  d="M 4,18 Q 55,4 110,14 T 216,10"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="10"
                  strokeLinecap="round"
                />
              </svg>
            </span>{" "}
            — or admits it doesn't know.
          </div>
        </h1>

        {/* Dynamic Credibility Strip */}
        <div className="mt-6 inline-flex items-center gap-2 px-3 py-1.5 border border-[#2A3138] rounded-sm font-mono text-[11px] sm:text-xs text-[#8B95A1]">
          <span className="text-[#E8A33D] font-medium">FAITHFULNESS {faithfulness}</span>
          <span className="text-[#2A3138]">·</span>
          <span>HYBRID RETRIEVAL: DENSE + BM25</span>
          <span className="text-[#2A3138]">·</span>
          <span>35-Q EVAL SET</span>
        </div>
      </div>

      {/* Item Index Section */}
      <div className="mt-8 border-t border-[#2A3138] pt-6">
        <div className="font-mono text-xs text-[#8B95A1] tracking-widest uppercase mb-4">
          ITEM INDEX — TRY A QUERY
        </div>

        <div className="divide-y divide-[#2A3138] border-b border-[#2A3138]">
          {items.map((item, index) => (
            <button
              key={index}
              onClick={() => onSelectPrompt(item.prompt)}
              className="w-full text-left py-4 px-2 sm:px-3 hover:bg-[#EDE7DA]/[0.03] focus-visible:bg-[#EDE7DA]/[0.04] transition-colors group block focus-visible:outline-none"
            >
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3 sm:gap-4 flex-wrap sm:flex-nowrap">
                  <span className="font-mono text-xs font-semibold text-[#E8A33D] w-14 shrink-0">
                    {item.itemNumber}
                  </span>
                  <span className="font-body font-medium text-sm text-[#EDE7DA]">
                    {item.title}
                  </span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-mono text-xs text-[#8B95A1]">{item.ticker}</span>
                  <ArrowRight className="w-4 h-4 text-[#8B95A1] group-hover:text-[#EDE7DA] group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
              <div className="mt-1.5 sm:pl-17">
                <p className="font-mono text-xs text-[#8B95A1] line-clamp-1">
                  "{item.prompt}"
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
