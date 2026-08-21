"use client";

import { ShieldCheck, RefreshCw, AlertTriangle, Loader2, FileSearch, Sparkles, CheckCircle2 } from "lucide-react";
import { CriticStatus } from "@/lib/useVerityChat";

interface CriticBadgeProps {
  statusNode?: string;
  criticStatus?: CriticStatus;
  isStreaming?: boolean;
}

export function CriticBadge({ statusNode, criticStatus, isStreaming }: CriticBadgeProps) {
  // If streaming and node is actively running
  if (isStreaming && statusNode && statusNode !== "completed" && statusNode !== "failed") {
    const nodeLabels: Record<string, { label: string; icon: any }> = {
      classifier: { label: "Classifying query intent & filters...", icon: Sparkles },
      retriever: { label: "Searching SEC 10-K vector & BM25 index...", icon: FileSearch },
      draft: { label: "Synthesizing answer from retrieved chunks...", icon: Loader2 },
      critic: { label: "Auditing groundedness with Critic node...", icon: ShieldCheck },
      finalizer: { label: "Finalizing response...", icon: CheckCircle2 },
    };

    const current = nodeLabels[statusNode] || { label: `Processing node '${statusNode}'...`, icon: Loader2 };
    const Icon = current.icon;

    return (
      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 text-xs font-medium animate-pulse">
        <Icon className="w-3.5 h-3.5 animate-spin" />
        <span>{current.label}</span>
      </div>
    );
  }

  // If critic status is available
  if (criticStatus) {
    if (criticStatus.grounded && criticStatus.retry_count === 0) {
      return (
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-semibold">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>Verified • 100% Grounded</span>
        </div>
      );
    }

    if (criticStatus.grounded && criticStatus.retry_count > 0) {
      return (
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs font-semibold">
          <RefreshCw className="w-3.5 h-3.5 text-amber-400" />
          <span>Verified • Self-Corrected ({criticStatus.retry_count} retry{criticStatus.retry_count > 1 ? "s" : ""})</span>
        </div>
      );
    }

    if (!criticStatus.grounded) {
      return (
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-rose-500/10 text-rose-400 border border-rose-500/20 text-xs font-semibold">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
          <span>Unverified Claims Detected ({criticStatus.unsupported_claims.length} flagged)</span>
        </div>
      );
    }
  }

  if (statusNode === "completed") {
    return (
      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-semibold">
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
        <span>Verified Response</span>
      </div>
    );
  }

  return null;
}
