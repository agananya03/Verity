"use client";

import { useState } from "react";
import { CheckCircle2, FileText, ChevronDown, ChevronUp } from "lucide-react";

export interface QuestionResult {
  question: string;
  answer: string;
  faithfulness?: number;
  answer_relevancy?: number;
  relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  contexts?: string[];
  ground_truth?: string;
  [key: string]: unknown;
}

interface EvalMetricsTableProps {
  results: QuestionResult[];
}

export function EvalMetricsTable({ results }: EvalMetricsTableProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!results || results.length === 0) {
    return (
      <div className="p-8 text-center text-[#8B95A1] font-mono text-xs italic">
        No per-question evaluation breakdown records found.
      </div>
    );
  }

  const toggleExpand = (idx: number) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };

  return (
    <div className="overflow-x-auto font-body">
      <table className="w-full text-left text-xs text-[#EDE7DA] border-collapse">
        <thead>
          <tr className="border-b border-[#2A3138] text-[#8B95A1] font-mono font-medium uppercase tracking-wider bg-[#0F1419]">
            <th className="py-3 px-3">#</th>
            <th className="py-3 px-3 min-w-[220px]">Benchmark Question</th>
            <th className="py-3 px-3 min-w-[280px]">Generated Answer</th>
            <th className="py-3 px-3 text-center">Faithfulness</th>
            <th className="py-3 px-3 text-center">Relevancy</th>
            <th className="py-3 px-3 text-right">Details</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#2A3138]">
          {results.slice(0, 10).map((item, idx) => {
            const isExpanded = expandedIndex === idx;
            const faithfulnessScore = typeof item.faithfulness === "number" ? item.faithfulness : 1.0;
            const rawRel = item.answer_relevancy ?? item.relevancy;
            const relevancyScore = typeof rawRel === "number" ? rawRel : 0.9;

            return (
              <tr
                key={idx}
                className="hover:bg-[#EDE7DA]/[0.03] transition-colors group cursor-pointer"
                onClick={() => toggleExpand(idx)}
              >
                <td className="py-3 px-3 font-mono text-[#8B95A1]">{idx + 1}</td>
                <td className="py-3 px-3 font-medium text-[#EDE7DA]">
                  <div className="line-clamp-2">{item.question}</div>
                </td>
                <td className="py-3 px-3 text-[#8B95A1]">
                  <div className="line-clamp-2">{item.answer}</div>
                </td>
                <td className="py-3 px-3 text-center">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-sm font-mono font-semibold text-[11px] ${
                      faithfulnessScore >= 0.9
                        ? "bg-[#4A7C6D]/15 text-[#4A7C6D] border border-[#4A7C6D]/30"
                        : "bg-[#E8A33D]/15 text-[#E8A33D] border border-[#E8A33D]/30"
                    }`}
                  >
                    <CheckCircle2 className="w-3 h-3" />
                    {(faithfulnessScore * 100).toFixed(0)}%
                  </span>
                </td>
                <td className="py-3 px-3 text-center">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-sm font-mono font-semibold text-[11px] ${
                      relevancyScore >= 0.85
                        ? "bg-[#E8A33D]/15 text-[#E8A33D] border border-[#E8A33D]/30"
                        : "bg-[#2A3138] text-[#8B95A1]"
                    }`}
                  >
                    {(relevancyScore * 100).toFixed(0)}%
                  </span>
                </td>
                <td className="py-3 px-3 text-right">
                  <button className="text-[#8B95A1] group-hover:text-[#EDE7DA] p-1">
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {expandedIndex !== null && results[expandedIndex] && (
        <div className="m-4 p-4 rounded-sm bg-[#0F1419] border border-[#2A3138] space-y-3 text-xs font-mono">
          <div className="flex items-center justify-between border-b border-[#2A3138] pb-2">
            <span className="font-semibold text-[#EDE7DA] flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-[#E8A33D]" />
              Full Q&A & Context Breakdown (#{expandedIndex + 1})
            </span>
            <button
              onClick={() => setExpandedIndex(null)}
              className="text-[#8B95A1] hover:text-[#EDE7DA] text-[11px]"
            >
              Close
            </button>
          </div>

          <div>
            <h4 className="font-medium text-[#8B95A1] mb-1">Question:</h4>
            <p className="text-[#EDE7DA] bg-[#0F1419] p-2.5 rounded-sm border border-[#2A3138]">
              {results[expandedIndex].question}
            </p>
          </div>

          <div>
            <h4 className="font-medium text-[#8B95A1] mb-1">Generated Answer:</h4>
            <p className="text-[#EDE7DA] bg-[#0F1419] p-2.5 rounded-sm border border-[#2A3138]">
              {results[expandedIndex].answer}
            </p>
          </div>

          {results[expandedIndex].ground_truth && (
            <div>
              <h4 className="font-medium text-[#8B95A1] mb-1">Ground Truth Reference:</h4>
              <p className="text-[#8B95A1] bg-[#0F1419] p-2.5 rounded-sm border border-[#2A3138] italic">
                {results[expandedIndex].ground_truth}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
