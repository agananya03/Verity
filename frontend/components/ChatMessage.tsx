"use client";

import { AlertCircle, AlertTriangle, RotateCcw } from "lucide-react";
import { Message } from "@/lib/useVerityChat";
import { CriticBadge } from "@/components/CriticBadge";
import { SourceCitations } from "@/components/SourceCitations";

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
  onRetry?: () => void;
}

export function ChatMessage({ message, isStreaming, onRetry }: ChatMessageProps) {
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
              isDraft={message.isDraft}
              isVerified={message.isVerified}
            />
          </div>

          {/* Draft indicator badge when answer is draft undergoing verification */}
          {message.isDraft && !message.isVerified && (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-sm bg-[#E8A33D]/10 border border-[#E8A33D]/30 text-[#E8A33D] font-mono text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#E8A33D] animate-ping" />
              <span>Draft synthesis — verification in progress against SEC filings...</span>
            </div>
          )}

          {/* Warning banner: ungrounded claims removed / cold start warning */}
          {message.warningNote && (
            <div
              className={`p-3 rounded-sm font-mono text-xs flex items-center gap-2 ${
                message.finalGrounded === false
                  ? "bg-[#C0503F]/10 border border-[#C0503F]/30 text-[#EDE7DA]"
                  : "bg-[#E8A33D]/10 border border-[#E8A33D]/30 text-[#EDE7DA]"
              }`}
            >
              {message.finalGrounded === false ? (
                <AlertTriangle className="w-4 h-4 text-[#C0503F] shrink-0" />
              ) : (
                <AlertCircle className="w-4 h-4 text-[#E8A33D] shrink-0" />
              )}
              <span>{message.warningNote}</span>
            </div>
          )}

          {/* Answer Text */}
          <div
            className={`font-body text-xs sm:text-sm leading-relaxed whitespace-pre-wrap ${
              message.isDraft && !message.isVerified ? "text-[#EDE7DA]/80 italic" : "text-[#EDE7DA]"
            }`}
          >
            {message.content ? (
              message.content
            ) : isStreaming ? (
              <span className="text-[#8B95A1] font-body text-xs flex items-center gap-1.5 not-italic">
                <span className="font-mono text-[#E8A33D]">{">"}</span>
                Retrieving SEC filings and formulating verified answer...
              </span>
            ) : (
              <span className="font-mono text-xs text-[#8B95A1] not-italic">No answer generated.</span>
            )}
          </div>

          {/* Error Failure State with Retry Button */}
          {message.isError && (
            <div className="pt-2 flex items-center gap-3">
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-sm bg-[#C0503F]/15 hover:bg-[#C0503F]/25 border border-[#C0503F]/40 text-[#EDE7DA] font-mono text-xs transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-[#C0503F]" />
                  <span>Retry Query</span>
                </button>
              )}
              {message.requestId && (
                <span className="font-mono text-[10px] text-[#8B95A1]">
                  ID: {message.requestId.slice(0, 8)}
                </span>
              )}
            </div>
          )}

          {/* Cited SEC Sources Drawer */}
          {message.citations && message.citations.length > 0 && (
            <SourceCitations citations={message.citations} />
          )}
        </div>
      </div>
    </div>
  );
}
