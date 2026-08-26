"use client";

import { CriticStatus } from "@/lib/useVerityChat";

interface CriticBadgeProps {
  statusNode?: string;
  criticStatus?: CriticStatus;
  isStreaming?: boolean;
}

export function CriticBadge({ statusNode, criticStatus, isStreaming }: CriticBadgeProps) {
  // If streaming and node is actively running -> terminal log-line output
  if (isStreaming && statusNode && statusNode !== "completed" && statusNode !== "failed") {
    const nodeLabels: Record<string, string> = {
      classifier: "Classifying query intent & filters...",
      retriever: "Searching SEC 10-K vector & BM25 index...",
      draft: "Synthesizing answer from retrieved chunks...",
      critic: "Auditing groundedness with Critic node...",
      finalizer: "Finalizing response...",
    };

    const label = nodeLabels[statusNode] || `Processing node '${statusNode}'...`;

    return (
      <div className="font-mono text-xs text-[#E8A33D] flex items-center gap-1.5">
        <span>{label}</span>
        <span className="animate-pulse font-bold">_</span>
      </div>
    );
  }

  // If critic status is available
  if (criticStatus) {
    if (criticStatus.is_decline) {
      return (
        <div className="font-mono text-xs text-[#8B95A1] flex items-center gap-1.5 uppercase tracking-wider font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-[#8B95A1]" />
          <span>NO MATCH IN CORPUS</span>
        </div>
      );
    }

    if (criticStatus.grounded && criticStatus.retry_count === 0) {
      return (
        <div className="font-mono text-xs text-[#4A7C6D] flex items-center gap-1.5 uppercase tracking-wider font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-[#4A7C6D]" />
          <span>GROUNDED</span>
        </div>
      );
    }

    if (criticStatus.grounded && criticStatus.retry_count > 0) {
      return (
        <div className="font-mono text-xs text-[#E8A33D] flex items-center gap-1.5 uppercase tracking-wider font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-[#E8A33D]" />
          <span>RETRIED {criticStatus.retry_count}x</span>
        </div>
      );
    }

    if (!criticStatus.grounded) {
      return (
        <div className="font-mono text-xs text-[#C0503F] flex items-center gap-1.5 uppercase tracking-wider font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-[#C0503F]" />
          <span>UNRESOLVED AFTER RETRY</span>
        </div>
      );
    }
  }

  if (statusNode === "completed") {
    return (
      <div className="font-mono text-xs text-[#4A7C6D] flex items-center gap-1.5 uppercase tracking-wider font-medium">
        <span className="w-1.5 h-1.5 rounded-full bg-[#4A7C6D]" />
        <span>GROUNDED</span>
      </div>
    );
  }

  return null;
}
