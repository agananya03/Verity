"use client";

import { AlertCircle } from "lucide-react";
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
    const showRewritten =
      message.rewrittenQuery &&
      message.rewrittenQuery.trim().toLowerCase() !== message.content.trim().toLowerCase();

    return (
      <div className="flex flex-col items-end my-4 space-y-1">
        <div className="flex items-start gap-2.5 max-w-[85%] sm:max-w-[75%]">
          {/* User Typed Note on Paper */}
          <div className="p-3.5 rounded-sm bg-[#EDE7DA] text-[#0F1419] font-body text-xs sm:text-sm leading-relaxed border border-[#EDE7DA]">
            {message.content}
          </div>
          {/* Square Mono Tag for User */}
          <div className="w-7 h-7 rounded-sm bg-[#0F1419] border border-[#2A3138] flex items-center justify-center shrink-0 font-mono text-xs text-[#8B95A1] mt-0.5">
            U
          </div>
        </div>

        {showRewritten && (
          <div className="font-mono text-[11px] text-[#8B95A1] pr-9 text-right">
            Interpreted as: &ldquo;{message.rewrittenQuery}&rdquo;
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex justify-start my-5">
      <div className="flex items-start gap-3 max-w-[92%] sm:max-w-[85%] w-full">
        {/* Square Mono Tag for Assistant */}
        <div className="w-7 h-7 rounded-sm bg-[#0F1419] border border-[#2A3138] flex items-center justify-center shrink-0 font-mono text-xs font-semibold text-[#E8A33D] mt-0.5">
          V
        </div>

        <div className="flex-1 p-4 rounded-sm bg-[#0F1419] border border-[#2A3138] space-y-3">
          {/* Header Bar with Assistant Label & Critic Badge */}
          <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-[#2A3138]">
            <span className="font-body text-xs font-medium text-[#8B95A1]">
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
            <div className="p-3 rounded-sm bg-[#E8A33D]/10 border border-[#E8A33D]/30 text-[#EDE7DA] font-mono text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-[#E8A33D] shrink-0" />
              <span>{message.warningNote}</span>
            </div>
          )}

          {/* Answer Text */}
          <div className="font-body text-xs sm:text-sm text-[#EDE7DA] leading-relaxed whitespace-pre-wrap">
            {message.content ? (
              message.content
            ) : isStreaming ? (
              <span className="text-[#8B95A1] font-body text-xs flex items-center gap-1.5">
                <span className="font-mono text-[#E8A33D]">{">"}</span>
                Retrieving SEC filings and formulating verified answer...
              </span>
            ) : (
              <span className="font-mono text-xs text-[#8B95A1]">No answer generated.</span>
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
