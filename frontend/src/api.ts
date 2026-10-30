import type {
  Character,
  CharacterCard,
  Health,
  ProviderSettings,
  Session,
  SessionMemory,
  StructuredMemoryList,
  StructuredMemoryRecord,
  StructuredMemoryStatus,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : detail && typeof detail === "object" && "message" in detail
      ? String((detail as { message: unknown }).message)
      : `请求失败（${status}）`);
    this.status = status;
    this.detail = detail;
  }
}

async function parseError(response: Response): Promise<never> {
  let detail: unknown = null;
  try {
    const body = await response.json();
    detail = body.detail ?? body;
  } catch {
    detail = response.statusText;
  }
  throw new ApiError(response.status, detail);
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof Blob) ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) await parseError(response);
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<Health> {
  const response = await fetch("/health");
  const body = await response.json();
  if (!response.ok && response.status !== 503) throw new ApiError(response.status, body);
  return body as Health;
}

export const getCharacters = () =>
  api<{ characters: Character[] }>("/characters").then((body) => body.characters);

export const getCharacterCard = (characterId: string) =>
  api<{ card: CharacterCard }>(`/characters/${encodeURIComponent(characterId)}`).then(
    (body) => body.card,
  );

export const deleteCharacter = (characterId: string) =>
  api<{ deleted: boolean; id: string }>(
    `/characters/${encodeURIComponent(characterId)}`,
    { method: "DELETE" },
  );

export const getProviderSettings = () =>
  api<ProviderSettings>("/settings/providers");

export const saveDeepSeekKey = (apiKey: string) =>
  api<{ saved: boolean; configured: boolean }>("/settings/deepseek-key", {
    method: "PUT",
    body: JSON.stringify({ api_key: apiKey }),
  });

export const saveOpenCodeKey = (apiKey: string) =>
  api<{ saved: boolean; configured: boolean }>("/settings/opencode-key", {
    method: "PUT",
    body: JSON.stringify({ api_key: apiKey }),
  });

export const deleteOpenCodeKey = () =>
  api<{ deleted: boolean; configured: boolean }>("/settings/opencode-key", {
    method: "DELETE",
  });

export const deleteDeepSeekKey = () =>
  api<{ deleted: boolean; configured: boolean }>("/settings/deepseek-key", {
    method: "DELETE",
  });

export const saveTokenLimits = (requestTokens: number, dailyTokens: number) =>
  api("/settings/token-limits", {
    method: "PUT",
    body: JSON.stringify({
      request_token_limit: requestTokens,
      daily_token_limit: dailyTokens,
    }),
  });

export const getRoleConversation = (
  characterId: string,
  provider: "opencode" | "deepseek" | "local" = "opencode",
) =>
  api<Session>(
    `/characters/${encodeURIComponent(characterId)}/conversation?provider=${provider}`,
    { method: "PUT" },
  );

export const resetRoleConversation = (characterId: string) =>
  api<Session>(
    `/characters/${encodeURIComponent(characterId)}/conversation/messages`,
    { method: "DELETE" },
  );

export const getSession = (sessionId: string) =>
  api<Session>(`/sessions/${sessionId}`);

export const getMemory = (sessionId: string) =>
  api<SessionMemory>(`/sessions/${sessionId}/memory`);

export const saveMemory = (
  sessionId: string,
  body: {
    notes: string;
    auto_enabled: boolean;
    revision: number;
    kept_items?: number[];
  },
) =>
  api<SessionMemory>(`/sessions/${sessionId}/memory`, {
    method: "PUT",
    body: JSON.stringify(body),
  });

export const getStructuredMemories = (characterId: string, includeInactive = true) =>
  api<StructuredMemoryList>(
    `/characters/${encodeURIComponent(characterId)}/structured-memory?include_inactive=${includeInactive}`,
  );

export const getStructuredMemoryStatus = (characterId: string) =>
  api<StructuredMemoryStatus>(
    `/characters/${encodeURIComponent(characterId)}/structured-memory/status`,
  );

export const extractStructuredMemories = (characterId: string) =>
  api<{
    created: StructuredMemoryRecord[];
    duplicates: number[];
    superseded: StructuredMemoryRecord[];
    contested: StructuredMemoryRecord[];
    ignored: Array<Record<string, unknown>>;
    rejected: Array<{ reason: string }>;
    source_messages: number;
    usage: Record<string, unknown>;
  }>(`/characters/${encodeURIComponent(characterId)}/structured-memory/extract`, {
    method: "POST",
  });

export const reviewStructuredMemory = (
  characterId: string,
  body: {
    action: "activate" | "revise" | "delete";
    item_id: number;
    old_item_id?: number;
  },
) =>
  api<{ action: string; item: StructuredMemoryRecord }>(
    `/characters/${encodeURIComponent(characterId)}/structured-memory/review`,
    { method: "POST", body: JSON.stringify(body) },
  );

export const stopGeneration = (sessionId: string) =>
  api(`/sessions/${sessionId}/stop`, { method: "POST" });

export async function uploadImage(sessionId: string, file: File) {
  const response = await fetch(
    `/images?session_id=${encodeURIComponent(sessionId)}`,
    {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    },
  );
  if (!response.ok) await parseError(response);
  return response.json() as Promise<{
    id: string;
    mime_type: string;
    width: number;
    height: number;
    url: string;
  }>;
}

export async function transcribe(file: Blob): Promise<string> {
  const response = await fetch("/transcribe", {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
  if (!response.ok) await parseError(response);
  return (await response.json()) as string;
}

export async function synthesize(sessionId: string, messageId: number): Promise<string> {
  const response = await fetch(`/sessions/${sessionId}/speech`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId }),
  });
  if (!response.ok) await parseError(response);
  return URL.createObjectURL(await response.blob());
}

export interface StreamEvent {
  event: "start" | "delta" | "done" | "error";
  data: Record<string, unknown>;
}

export async function streamChat(
  payload: {
    session_id: string;
    message: string;
    image_id?: string;
    max_new_tokens: number;
    request_id: string;
    timeout_seconds: number;
  },
  onEvent: (event: StreamEvent) => void,
  signal: AbortSignal,
) {
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) await parseError(response);
  if (!response.body) throw new Error("浏览器未收到流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      onEvent({ event: event as StreamEvent["event"], data: JSON.parse(data) });
    }
  }
}
