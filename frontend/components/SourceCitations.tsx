"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Citation } from "@/lib/useVerityChat";

interface SourceCitationsProps {
  citations: Citation[];
}

export function SourceCitations({ citations }: SourceCitationsProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-4 border-t border-[#2A3138] pt-3 font-mono text-xs">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full flex items-center justify-between text-[#8B95A1] hover:text-[#EDE7DA] py-1 transition-colors group focus-visible:outline-none"
      >
        <div className="flex items-center gap-2">
          <span className="text-[#E8A33D] font-semibold">SOURCES</span>
          <span className="text-[11px] text-[#8B95A1]">({citations.length})</span>
        </div>
        <div className="flex items-center gap-1 text-[11px] text-[#8B95A1] group-hover:text-[#EDE7DA]">
          <span>{isOpen ? "HIDE" : "VIEW"}</span>
          {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </div>
      </button>

      {isOpen && (
        <div className="mt-3 space-y-2 max-h-72 overflow-y-auto pr-1">
          {citations.map((cite, index) => {
            const companyStr = cite.company || "SEC FILING";
            const yearStr = cite.fiscal_year ? `FY${cite.fiscal_year}` : "";
            const sectionStr = cite.item_section || "";
            const metaHeader = [companyStr, yearStr, sectionStr].filter(Boolean).join(" · ");

            return (
              <div
                key={cite.chunk_id || index}
                className="p-3 rounded-sm bg-[#0F1419] border border-[#2A3138] space-y-2 hover:border-[#8B95A1]/40 transition-colors"
              >
                <div className="flex items-center justify-between gap-2 text-[11px]">
                  <div className="font-mono text-[#E8A33D] font-medium tracking-wide">
                    <span className="bg-[#E8A33D]/15 text-[#E8A33D] px-1.5 py-0.5 rounded-sm">
                      {metaHeader}
                    </span>
                  </div>
                  <span className="text-[10px] text-[#8B95A1] font-mono">
                    CHUNK #{index + 1}
                  </span>
                </div>

                {cite.text && (
                  <p className="font-body text-[#8B95A1] text-xs leading-relaxed line-clamp-3 bg-[#0F1419] p-2 rounded-sm border border-[#2A3138]/60">
                    "{cite.text.trim()}"
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
