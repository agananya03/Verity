"use client";

import { useEffect, useState } from "react";
import { Database } from "lucide-react";

interface CompanyInfo {
  ticker: string;
  name: string;
  fiscal_years: string[];
}

interface CorpusInfoResponse {
  companies: CompanyInfo[];
  filing_type: string;
  total_chunks: number;
}

export function CoverageIndicator() {
  const [corpusInfo, setCorpusInfo] = useState<CorpusInfoResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function fetchCorpusInfo() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${apiUrl}/corpus-info`);
        if (res.ok) {
          const data = await res.json();
          setCorpusInfo(data);
        }
      } catch (err) {
        console.warn("Could not fetch corpus info:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchCorpusInfo();
  }, []);

  if (loading) {
    return (
      <div className="font-mono text-[11px] text-[#8B95A1] flex items-center gap-1.5 animate-pulse">
        <span className="w-1.5 h-1.5 rounded-full bg-[#E8A33D]" />
        <span>Loading corpus coverage…</span>
      </div>
    );
  }

  if (!corpusInfo || !corpusInfo.companies || corpusInfo.companies.length === 0) {
    return (
      <div className="font-mono text-[11px] text-[#8B95A1] flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-[#8B95A1]" />
        <span>INDEXED: SEC Form 10-K Corpus</span>
      </div>
    );
  }

  const tickersStr = corpusInfo.companies.map((c) => c.ticker).join(" · ");

  // Aggregate all fiscal years across all companies to compute min and max range
  const allYears = corpusInfo.companies
    .flatMap((c) => c.fiscal_years)
    .map((y) => parseInt(y, 10))
    .filter((y) => !isNaN(y));

  let yearRangeStr = "";
  if (allYears.length > 0) {
    const minYear = Math.min(...allYears);
    const maxYear = Math.max(...allYears);
    yearRangeStr = minYear === maxYear ? `FY${minYear}` : `FY${minYear}-${maxYear}`;
  }

  return (
    <div className="font-mono text-[11px] text-[#8B95A1] flex items-center gap-1.5 tracking-normal overflow-x-auto whitespace-nowrap">
      <Database className="w-3 h-3 text-[#E8A33D] shrink-0" />
      <span className="text-[#EDE7DA] font-medium shrink-0">INDEXED:</span>
      <span className="text-[#EDE7DA]/90">{tickersStr}</span>
      {yearRangeStr && (
        <>
          <span className="text-[#2A3138]">—</span>
          <span className="text-[#E8A33D]">{yearRangeStr}</span>
        </>
      )}
    </div>
  );
}
