"use client";

import { useState, useCallback, useRef } from "react";

export interface Citation {
  chunk_id: string;
  company: string;
  fiscal_year: string;
  item_section: string;
  text?: string;
  [key: string]: any;
}

export interface CriticStatus {
  grounded: boolean;
  unsupported_claims: string[];
  retry_count: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  criticStatus?: CriticStatus;
  statusNode?: string;
  isError?: boolean;
  warningNote?: string;
}

export function useVerityChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (query: string) => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    setError(null);
    setIsStreaming(true);

    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now()}`;

    const userMessage: Message = {
      id: userMessageId,
      role: "user",
      content: trimmedQuery,
    };

    const initialAssistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      citations: [],
      criticStatus: undefined,
      statusNode: "classifier",
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);

    // Setup abort controller for current stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    // Timer to detect backend cold-start (Render/HuggingFace free tier spin-up)
    const coldStartTimer = setTimeout(() => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                warningNote:
                  "Waking up the backend (free-tier hosting can take up to a minute on first request)...",
              }
            : msg
        )
      );
    }, 6000);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmedQuery }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error("Response body is null or streaming is not supported.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      let eventType = "message";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // First chunk received -> clear cold start warning
        clearTimeout(coldStartTimer);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId ? { ...msg, warningNote: undefined } : msg
          )
        );

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) {
            eventType = "message";
            continue;
          }

          if (trimmedLine.startsWith("event:")) {
            eventType = trimmedLine.replace(/^event:\s*/, "").trim();
          } else if (trimmedLine.startsWith("data:")) {
            const dataStr = trimmedLine.replace(/^data:\s*/, "").trim();

            try {
              const data = JSON.parse(dataStr);

              setMessages((prev) =>
                prev.map((msg) => {
                  if (msg.id !== assistantMessageId) return msg;

                  const updated = { ...msg };

                  switch (eventType) {
                    case "status":
                      if (data.node) {
                        updated.statusNode = data.node;
                      }
                      break;

                    case "citation":
                      if (Array.isArray(data)) {
                        updated.citations = data;
                      }
                      break;

                    case "answer":
                      if (data.answer) {
                        updated.content = data.answer;
                      }
                      break;

                    case "critic_status":
                      updated.criticStatus = {
                        grounded: Boolean(data.grounded),
                        unsupported_claims: Array.isArray(data.unsupported_claims)
                          ? data.unsupported_claims
                          : [],
                        retry_count: Number(data.retry_count || 0),
                      };
                      break;

                    case "done":
                      if (data.final_answer) {
                        updated.content = data.final_answer;
                      }
                      updated.statusNode = "completed";
                      break;

                    case "error":
                      updated.isError = true;
                      updated.content = data.error || "An error occurred during graph processing.";
                      break;

                    default:
                      break;
                  }

                  return updated;
                })
              );
            } catch (err) {
              console.warn("Failed to parse SSE data JSON:", dataStr, err);
            }
          }
        }
      }
    } catch (err: any) {
      clearTimeout(coldStartTimer);

      if (err.name === "AbortError") {
        console.log("Chat stream aborted by user.");
        return;
      }

      console.error("Error in useVerityChat stream:", err);

      const errorMessage =
        err.message || "Unable to connect to Verity backend. Please ensure the backend server is running.";

      setError(errorMessage);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                isError: true,
                content:
                  msg.content ||
                  "Unable to complete query. Please check if backend server is running at " +
                    apiUrl,
                statusNode: "failed",
                warningNote: undefined,
              }
            : msg
        )
      );
    } finally {
      clearTimeout(coldStartTimer);
      setIsStreaming(false);
    }
  }, []);

  const clearMessages = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setMessages([]);
    setError(null);
    setIsStreaming(false);
  }, []);

  return {
    messages,
    sendMessage,
    isStreaming,
    error,
    clearMessages,
  };
}
