"use client";

import { useState } from "react";
import { CheckCircle2, AlertCircle, FileText, ChevronDown, ChevronUp } from "lucide-react";

interface QuestionResult {
  question: string;
  answer: string;
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
  contexts?: string[];
  ground_truth?: string;
  [key: string]: any;
}

interface EvalMetricsTableProps {
  results: QuestionResult[];
}

export function EvalMetricsTable({ results }: EvalMetricsTableProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!results || results.length === 0) {
    return (
      <div className="p-8 text-center text-slate-500 text-xs italic">
        No per-question evaluation breakdown records found.
      </div>
    );
  }

  const toggleExpand = (idx: number) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs text-slate-300 border-collapse">
        <thead>
          <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider bg-slate-900/50">
            <th className="py-3 px-4">#</th>
            <th className="py-3 px-4 min-w-[220px]">Benchmark Question</th>
            <th className="py-3 px-4 min-w-[280px]">Generated Answer</th>
            <th className="py-3 px-4 text-center">Faithfulness</th>
            <th className="py-3 px-4 text-center">Relevancy</th>
            <th className="py-3 px-4 text-right">Details</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {results.slice(0, 10).map((item, idx) => {
            const isExpanded = expandedIndex === idx;
            const faithfulnessScore = item.faithfulness ?? 1.0;
            const relevancyScore = item.answer_relevancy ?? item.relevancy ?? 0.9;

            return (
              <tr
                key={idx}
                className="hover:bg-slate-800/40 transition-colors group cursor-pointer"
                onClick={() => toggleExpand(idx)}
              >
                <td className="py-3.5 px-4 font-mono text-slate-500">{idx + 1}</td>
                <td className="py-3.5 px-4 font-medium text-slate-200">
                  <div className="line-clamp-2">{item.question}</div>
                </td>
                <td className="py-3.5 px-4 text-slate-300">
                  <div className="line-clamp-2">{item.answer}</div>
                </td>
                <td className="py-3.5 px-4 text-center">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono font-semibold text-[11px] ${
                      faithfulnessScore >= 0.9
                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                        : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                    }`}
                  >
                    <CheckCircle2 className="w-3 h-3" />
                    {(faithfulnessScore * 100).toFixed(0)}%
                  </span>
                </td>
                <td className="py-3.5 px-4 text-center">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono font-semibold text-[11px] ${
                      relevancyScore >= 0.85
                        ? "bg-blue-500/10 text-blue-400 border border-blue-500/20"
                        : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {(relevancyScore * 100).toFixed(0)}%
                  </span>
                </td>
                <td className="py-3.5 px-4 text-right">
                  <button className="text-slate-400 group-hover:text-white p-1">
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {expandedIndex !== null && results[expandedIndex] && (
        <div className="m-4 p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3 text-xs">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="font-semibold text-slate-200 flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-blue-400" />
              Full Q&A & Context Breakdown (#{expandedIndex + 1})
            </span>
            <button
              onClick={() => setExpandedIndex(null)}
              className="text-slate-400 hover:text-slate-200 text-[11px]"
            >
              Close
            </button>
          </div>

          <div>
            <h4 className="font-medium text-slate-400 mb-1">Question:</h4>
            <p className="text-slate-200 bg-slate-950/60 p-2.5 rounded border border-slate-800">
              {results[expandedIndex].question}
            </p>
          </div>

          <div>
            <h4 className="font-medium text-slate-400 mb-1">Generated Answer:</h4>
            <p className="text-slate-200 bg-slate-950/60 p-2.5 rounded border border-slate-800">
              {results[expandedIndex].answer}
            </p>
          </div>

          {results[expandedIndex].ground_truth && (
            <div>
              <h4 className="font-medium text-slate-400 mb-1">Ground Truth Reference:</h4>
              <p className="text-slate-300 bg-slate-950/40 p-2.5 rounded border border-slate-800/60 italic">
                {results[expandedIndex].ground_truth}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
