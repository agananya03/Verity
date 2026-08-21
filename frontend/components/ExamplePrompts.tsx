"use client";

import { Sparkles, ArrowRight, Building2, TrendingUp, ShieldAlert, HelpCircle } from "lucide-react";

interface ExamplePromptsProps {
  onSelectPrompt: (prompt: string) => void;
}

export function ExamplePrompts({ onSelectPrompt }: ExamplePromptsProps) {
  const examples = [
    {
      title: "R&D Spend Analysis",
      prompt: "What was Apple's total R&D spend in fiscal year 2024?",
      icon: Building2,
      category: "Apple 10-K",
      color: "from-blue-500/20 to-indigo-500/20 text-blue-400 border-blue-500/30",
    },
    {
      title: "Segment Revenue Comparison",
      prompt: "Compare Microsoft's Intelligent Cloud revenue between fiscal year 2023 and 2024.",
      icon: TrendingUp,
      category: "Microsoft 10-K",
      color: "from-emerald-500/20 to-teal-500/20 text-emerald-400 border-emerald-500/30",
    },
    {
      title: "Risk Factors Audit",
      prompt: "What are the primary supply chain risk factors listed in Apple's Item 1A filing?",
      icon: ShieldAlert,
      category: "Risk Analysis",
      color: "from-amber-500/20 to-orange-500/20 text-amber-400 border-amber-500/30",
    },
    {
      title: "Unanswerable Query Test",
      prompt: "What was Netflix's global subscriber churn rate and ARPU in Q4 2024?",
      icon: HelpCircle,
      category: "Negative Test",
      color: "from-purple-500/20 to-pink-500/20 text-purple-400 border-purple-500/30",
    },
  ];

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] py-8 text-center max-w-3xl mx-auto px-4">
      {/* Hero Header */}
      <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-emerald-400 p-0.5 shadow-xl shadow-blue-500/20 mb-5">
        <div className="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center">
          <Sparkles className="w-7 h-7 text-emerald-400" />
        </div>
      </div>

      <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-100 to-slate-400">
        Financial Intelligence RAG
      </h1>

      <p className="mt-2.5 text-slate-400 text-sm sm:text-base max-w-lg leading-relaxed">
        Query SEC 10-K filings with self-correcting Critic node validation & hybrid dense/sparse retrieval.
      </p>

      {/* Grid of Interactive Sample Questions */}
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full text-left">
        {examples.map((item, i) => {
          const Icon = item.icon;
          return (
            <button
              key={i}
              onClick={() => onSelectPrompt(item.prompt)}
              className="p-4 rounded-2xl glass-panel border border-slate-800/80 hover:border-slate-700 transition-all duration-300 group flex flex-col justify-between hover:shadow-lg hover:shadow-blue-500/5 hover:-translate-y-0.5"
            >
              <div className="flex items-center justify-between w-full mb-2">
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-gradient-to-r ${item.color} border`}>
                  <Icon className="w-3 h-3" />
                  {item.category}
                </span>
                <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-slate-200 group-hover:translate-x-0.5 transition-all" />
              </div>
              <div>
                <h3 className="text-xs font-semibold text-slate-300 group-hover:text-white mb-1">
                  {item.title}
                </h3>
                <p className="text-xs text-slate-400 group-hover:text-slate-300 line-clamp-2 leading-relaxed">
                  "{item.prompt}"
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
