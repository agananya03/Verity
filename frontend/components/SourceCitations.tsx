"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, FileText, ExternalLink, Building2, Calendar, Tag } from "lucide-react";
import { Citation } from "@/lib/useVerityChat";

interface SourceCitationsProps {
  citations: Citation[];
}

export function SourceCitations({ citations }: SourceCitationsProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-4 border-t border-slate-800/60 pt-3">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full flex items-center justify-between text-xs font-medium text-slate-400 hover:text-slate-200 py-1 transition-colors group"
      >
        <div className="flex items-center gap-2">
          <FileText className="w-3.5 h-3.5 text-blue-400" />
          <span>Cited SEC 10-K Sources ({citations.length})</span>
        </div>
        <div className="flex items-center gap-1 text-[11px] text-slate-500 group-hover:text-slate-300">
          <span>{isOpen ? "Hide" : "View"}</span>
          {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </div>
      </button>

      {isOpen && (
        <div className="mt-3 space-y-2.5 max-h-72 overflow-y-auto pr-1">
          {citations.map((cite, index) => (
            <div
              key={cite.chunk_id || index}
              className="p-3 rounded-xl bg-slate-900/70 border border-slate-800 text-xs space-y-1.5 hover:border-slate-700 transition-colors"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
                <div className="flex items-center gap-2 font-medium">
                  <span className="flex items-center gap-1 text-slate-300">
                    <Building2 className="w-3 h-3 text-blue-400" />
                    {cite.company || "SEC Filing"}
                  </span>
                  <span className="flex items-center gap-1 text-slate-400">
                    <Calendar className="w-3 h-3 text-emerald-400" />
                    {cite.fiscal_year || "N/A"}
                  </span>
                  {cite.item_section && (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px]">
                      <Tag className="w-2.5 h-2.5 text-indigo-400" />
                      {cite.item_section}
                    </span>
                  )}
                </div>
                <span className="text-[10px] text-slate-500 font-mono">
                  Chunk #{index + 1}
                </span>
              </div>

              {cite.text && (
                <p className="text-slate-300 text-[11px] leading-relaxed line-clamp-3 italic bg-slate-950/40 p-2 rounded border border-slate-800/40">
                  "{cite.text.trim()}"
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
