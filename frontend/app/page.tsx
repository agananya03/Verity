"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Sparkles, Trash2, AlertCircle, Loader2 } from "lucide-react";
import { useVerityChat } from "@/lib/useVerityChat";
import { ChatMessage } from "@/components/ChatMessage";
import { ExamplePrompts } from "@/components/ExamplePrompts";

export default function ChatPage() {
  const { messages, sendMessage, isStreaming, error, clearMessages } = useVerityChat();
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

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    const query = input;
    setInput("");
    await sendMessage(query);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSelectExample = (prompt: string) => {
    setInput("");
    sendMessage(prompt);
  };

  return (
    <div className="flex flex-col min-h-[calc(100vh-5.5rem)] relative pb-28">
      {/* Top Action Bar when messages exist */}
      {messages.length > 0 && (
        <div className="flex items-center justify-between py-2 px-1 mb-4 border-b border-slate-800/60 text-xs">
          <div className="flex items-center gap-2 text-slate-400">
            <Sparkles className="w-3.5 h-3.5 text-blue-400" />
            <span>Active Financial Query Thread</span>
          </div>
          <button
            onClick={clearMessages}
            disabled={isStreaming}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800 transition-colors disabled:opacity-50"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Chat</span>
          </button>
        </div>
      )}

      {/* Main Messages Container / Empty State */}
      <div className="flex-1 space-y-4">
        {messages.length === 0 ? (
          <ExamplePrompts onSelectPrompt={handleSelectExample} />
        ) : (
          messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} isStreaming={isStreaming && msg.id === messages[messages.length - 1]?.id} />
          ))
        )}

        {/* Global Error Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center justify-between gap-3 shadow-lg">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Fixed/Sticky Bottom Input Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-40 bg-gradient-to-t from-[#0b0f19] via-[#0b0f19]/95 to-transparent pt-6 pb-4">
        <div className="max-w-4xl mx-auto px-4 sm:px-6">
          <div className="glass-panel p-2 rounded-2xl border border-slate-800 shadow-2xl focus-within:border-blue-500/50 transition-colors">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
              className="flex items-end gap-2"
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask any financial question about Apple, Microsoft SEC 10-Ks..."
                rows={1}
                disabled={isStreaming}
                className="flex-1 bg-transparent border-0 resize-none px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-0 max-h-32 min-h-[44px]"
              />

              <button
                type="submit"
                disabled={isStreaming || !input.trim()}
                className="w-10 h-10 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white flex items-center justify-center shrink-0 shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-40 disabled:hover:scale-100 disabled:shadow-none"
              >
                {isStreaming ? (
                  <Loader2 className="w-4 h-4 animate-spin text-blue-200" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </form>
          </div>
          <p className="text-[11px] text-center text-slate-500 mt-2">
            Verity RAG uses self-correcting Critic audit nodes to ground answers in verified SEC 10-K filings.
          </p>
        </div>
      </div>
    </div>
  );
}
