"use client";

import { User, ShieldCheck, AlertCircle } from "lucide-react";
import { Message } from "@/lib/useVerityChat";
import { CriticBadge } from "@/components/CriticBadge";
import { SourceCitations } from "@/components/SourceCitations";

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end my-4">
        <div className="flex items-start gap-3 max-w-[85%] sm:max-w-[75%]">
          <div className="p-4 rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 text-white shadow-lg shadow-blue-500/10 border border-blue-400/20 text-sm leading-relaxed font-medium">
            {message.content}
          </div>
          <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 text-slate-300">
            <User className="w-4 h-4" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start my-5">
      <div className="flex items-start gap-3 max-w-[92%] sm:max-w-[85%] w-full">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-emerald-400 p-0.5 shadow-md shadow-blue-500/20 shrink-0 mt-1">
          <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
        </div>

        <div className="flex-1 p-5 rounded-2xl glass-panel border border-slate-800/80 shadow-xl space-y-3">
          {/* Header Bar with Critic Badge */}
          <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-slate-800/50">
            <span className="text-xs font-semibold tracking-wide text-slate-300 flex items-center gap-1.5">
              Verity AI Assistant
            </span>
            <CriticBadge
              statusNode={message.statusNode}
              criticStatus={message.criticStatus}
              isStreaming={isStreaming}
            />
          </div>

          {/* Cold start warning notice */}
          {message.warningNote && (
            <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-center gap-2 animate-pulse">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{message.warningNote}</span>
            </div>
          )}

          {/* Answer Text */}
          <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap font-normal">
            {message.content ? (
              message.content
            ) : isStreaming ? (
              <span className="text-slate-400 italic flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-400 animate-ping" />
                Retrieving SEC filings and formulating verified answer...
              </span>
            ) : (
              <span className="text-slate-500 italic">No answer generated.</span>
            )}
          </div>

          {/* Cited SEC Sources Drawer */}
          {message.citations && message.citations.length > 0 && (
            <SourceCitations citations={message.citations} />
          )}
        </div>
      </div>
    </div>
  );
}
