"use client";

import { useState, useCallback, useRef, useEffect } from "react";

export interface Citation {
  chunk_id: string;
  citation_marker?: string;
  company: string;
  fiscal_year: string;
  item_section: string;
  item_title?: string;
  source_path?: string;
  source_url?: string;
  text?: string;
  [key: string]: unknown;
}

export interface CriticStatus {
  grounded: boolean;
  unsupported_claims: string[];
  retry_count: number;
  is_decline?: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  rewrittenQuery?: string;
  citations?: Citation[];
  criticStatus?: CriticStatus;
  statusNode?: string;
  isError?: boolean;
  warningNote?: string;
  isDraft?: boolean;
  isVerified?: boolean;
  finalGrounded?: boolean;
  unsupportedClaimsFinal?: string[];
  requestId?: string;
}

export function useVerityChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<Message[]>([]);
  const lastQueryRef = useRef<string>("");

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const sendMessage = useCallback(async (query: string) => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    lastQueryRef.current = trimmedQuery;
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
      statusNode: "rewriter",
      isDraft: false,
      isVerified: false,
    };

    // Extract recent conversation history (last 6 completed messages) from messagesRef.current
    const historyPayload = messagesRef.current
      .filter((msg) => msg.content && !msg.isError)
      .slice(-6)
      .map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

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
        body: JSON.stringify({
          query: trimmedQuery,
          history: historyPayload,
        }),
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
            continue;
          }

          if (trimmedLine.startsWith("event:")) {
            eventType = trimmedLine.replace(/^event:\s*/, "").trim();
          } else if (trimmedLine.startsWith("data:")) {
            const dataStr = trimmedLine.replace(/^data:\s*/, "").trim();

            try {
              const data = JSON.parse(dataStr);

              if (eventType === "rewritten_query" && data.rewritten_query) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === userMessageId
                      ? { ...msg, rewrittenQuery: data.rewritten_query }
                      : msg
                  )
                );
              }

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
                        // Draft state: Do not display as verified final answer
                        updated.content = data.answer;
                        updated.isDraft = true;
                        updated.isVerified = false;
                      }
                      break;

                    case "critic_status":
                      const isGrounded = Boolean(data.grounded);
                      updated.criticStatus = {
                        grounded: isGrounded,
                        unsupported_claims: Array.isArray(data.unsupported_claims)
                          ? data.unsupported_claims
                          : [],
                        retry_count: Number(data.retry_count || 0),
                        is_decline: Boolean(data.is_decline),
                      };
                      // If critic verifies grounded=true, transition out of draft
                      if (isGrounded) {
                        updated.isVerified = true;
                        updated.isDraft = false;
                      }
                      break;

                    case "done":
                      if (data.final_answer) {
                        updated.content = data.final_answer;
                      }
                      updated.statusNode = "completed";
                      updated.isDraft = false;
                      updated.isVerified = true;
                      if (data.request_id) {
                        updated.requestId = data.request_id;
                      }
                      const finalGrounded = typeof data.final_grounded === "boolean" ? data.final_grounded : true;
                      updated.finalGrounded = finalGrounded;
                      updated.unsupportedClaimsFinal = Array.isArray(data.unsupported_claims_final)
                        ? data.unsupported_claims_final
                        : [];
                      // If final_grounded is false in done, set visible warning banner
                      if (!finalGrounded) {
                        updated.warningNote = "Some claims could not be verified and were removed.";
                      }
                      break;

                    case "error":
                      updated.isError = true;
                      updated.isDraft = false;
                      updated.statusNode = "failed";
                      updated.content = data.error || "An error occurred during graph processing.";
                      if (data.request_id) {
                        updated.requestId = data.request_id;
                      }
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
    } catch (err: unknown) {
      clearTimeout(coldStartTimer);

      const errorObj = err instanceof Error ? err : new Error(String(err));
      if (errorObj.name === "AbortError") {
        return;
      }

      console.error("Error in useVerityChat stream:", errorObj);

      const errorMessage =
        errorObj.message || "Unable to connect to Verity backend. Please ensure the backend server is running.";

      setError(errorMessage);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
              ...msg,
              isError: true,
              isDraft: false,
              content:
                msg.content ||
                "Unable to complete query. The backend service may be cold-starting or unreachable. Click Retry to try again.",
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

  const retry = useCallback(() => {
    if (lastQueryRef.current) {
      sendMessage(lastQueryRef.current);
    }
  }, [sendMessage]);

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
    retry,
    isStreaming,
    error,
    clearMessages,
  };
}
