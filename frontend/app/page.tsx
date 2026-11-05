"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Trash2, AlertCircle, Loader2, RotateCcw } from "lucide-react";
import { useVerityChat } from "@/lib/useVerityChat";
import { ChatMessage } from "@/components/ChatMessage";
import { ExamplePrompts } from "@/components/ExamplePrompts";
import { CoverageIndicator } from "@/components/CoverageIndicator";

export default function ChatPage() {
  const { messages, sendMessage, retry, isStreaming, error, clearMessages } = useVerityChat();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages or streaming updates
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  const isSubmittingRef = useRef(false);

  const handleSend = async () => {
    if (!input.trim() || isStreaming || isSubmittingRef.current) return;
    isSubmittingRef.current = true;
    const query = input;
    setInput("");
    try {
      await sendMessage(query);
    } finally {
      isSubmittingRef.current = false;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSelectExample = (prompt: string) => {
    if (isStreaming || isSubmittingRef.current) return;
    setInput("");
    sendMessage(prompt);
  };

  return (
    <div className="flex flex-col min-h-[calc(100vh-5.5rem)] relative pb-32">
      {/* Top Action Bar when messages exist */}
      {messages.length > 0 && (
        <div className="flex items-center justify-between py-2 px-1 mb-4 border-b border-[#2A3138] font-mono text-xs text-[#8B95A1]">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#E8A33D]" />
            <span>ACTIVE FINANCIAL QUERY THREAD</span>
          </div>
          <button
            type="button"
            onClick={clearMessages}
            disabled={isStreaming}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-sm bg-[#0F1419] text-[#8B95A1] hover:text-[#EDE7DA] border border-[#2A3138] transition-colors disabled:opacity-50 cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Thread</span>
          </button>
        </div>
      )}

      {/* Main Messages Container / Empty State */}
      <div className="flex-1 space-y-4">
        {messages.length === 0 ? (
          <div className="space-y-6">
            <ExamplePrompts onSelectPrompt={handleSelectExample} />
          </div>
        ) : (
          messages.map((msg) => (
            <ChatMessage
              key={msg.id}
              message={msg}
              isStreaming={isStreaming && msg.id === messages[messages.length - 1]?.id}
              onRetry={retry}
            />
          ))
        )}

        {/* Global Error State for Connection / Cold-Start Failures with Retry */}
        {error && (
          <div className="p-4 rounded-sm bg-[#C0503F]/10 border border-[#C0503F]/30 text-[#EDE7DA] font-mono text-xs flex items-center justify-between gap-3 shadow-lg flex-wrap">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-[#C0503F] shrink-0" />
              <span>{error}</span>
            </div>
            <button
              type="button"
              onClick={retry}
              disabled={isStreaming}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-sm bg-[#C0503F]/20 hover:bg-[#C0503F]/30 border border-[#C0503F]/40 text-[#EDE7DA] font-mono text-xs transition-colors disabled:opacity-50 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5 text-[#C0503F]" />
              <span>Retry Query</span>
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Fixed/Sticky Bottom Input Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-40 bg-[#0F1419]/95 backdrop-blur border-t border-[#2A3138] pt-3 pb-4">
        <div className="max-w-4xl mx-auto px-4 sm:px-6">
          <div className="mb-2 px-0.5">
            <CoverageIndicator />
          </div>
          <div className="bg-[#0F1419] p-1.5 rounded-sm border border-[#2A3138] focus-within:border-[#E8A33D] transition-colors">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
              className="flex items-center gap-2"
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask any financial question about Apple, Microsoft, Amazon, NVIDIA 10-Ks…"
                rows={1}
                disabled={isStreaming}
                className="flex-1 bg-transparent border-0 resize-none px-3 py-2 font-mono text-xs sm:text-sm text-[#EDE7DA] placeholder:text-[#8B95A1] focus:outline-none focus:ring-0 max-h-32 min-h-[40px]"
              />

              <button
                type="submit"
                disabled={isStreaming || !input.trim()}
                title="Send query"
                className="w-9 h-9 rounded-sm bg-[#E8A33D] text-[#0F1419] flex items-center justify-center shrink-0 hover:bg-[#E8A33D]/90 active:scale-95 transition-all disabled:opacity-40 disabled:hover:scale-100 focus-visible:outline-none cursor-pointer"
              >
                {isStreaming ? (
                  <Loader2 className="w-4 h-4 animate-spin text-[#0F1419]" />
                ) : (
                  <Send className="w-4 h-4 text-[#0F1419]" />
                )}
              </button>
            </form>
          </div>
          <p className="font-mono text-[11px] text-center text-[#8B95A1] mt-2">
            Every answer is checked against its source before you see it.
          </p>
        </div>
      </div>
    </div>
  );
}
