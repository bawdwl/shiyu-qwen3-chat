import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  deleteCharacter,
  deleteDeepSeekKey,
  deleteOpenCodeKey,
  getCharacterCard,
  getCharacters,
  getHealth,
  getMemory,
  getProviderSettings,
  getRoleConversation,
  getStructuredMemories,
  getStructuredMemoryStatus,
  resetRoleConversation,
  reviewStructuredMemory,
  saveDeepSeekKey,
  saveOpenCodeKey,
  saveMemory,
  saveTokenLimits,
  stopGeneration,
  streamChat,
  synthesize,
  transcribe,
  uploadImage,
} from "./api";
import type {
  Character,
  CharacterCard,
  Health,
  Message,
  ProviderSettings,
  ProviderName,
  Session,
  SessionMemory,
  StructuredMemoryRecord,
  StructuredMemoryStatus,
} from "./types";

const PORTRAITS: Record<string, string> = {
  companion: "/ui/media/portraits/xiaoman.png",
  interview: "/ui/media/portraits/yanchuan.png",
  study: "/ui/media/portraits/zhixia.png",
};

type CloudProviderName = "opencode" | "deepseek";

function initialCloudProvider(): CloudProviderName {
  try {
    const stored = localStorage.getItem("shiyu-cloud-provider");
    return stored === "deepseek" ? "deepseek" : "opencode";
  } catch {
    return "opencode";
  }
}

function providerLabel(provider: CloudProviderName) {
  return provider === "opencode" ? "OpenCode Go" : "DeepSeek";
}

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    spark: "M12 2l1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8Z",
    chat: "M4 4h16v13H9l-5 4Z",
    settings: "M4 7h16M4 17h16M8 4v6m8 4v6",
    plus: "M12 5v14M5 12h14",
    image: "M4 5h16v14H4Zm2 11 4-4 3 3 2-2 3 3M8 9h.01",
    send: "M3 11.5 21 3l-8.5 18-2.2-7.3Z",
    stop: "M7 7h10v10H7Z",
    mic: "M9 5a3 3 0 0 1 6 0v6a3 3 0 0 1-6 0Zm-3 5a6 6 0 0 0 12 0M12 16v4m-4 0h8",
    volume: "M3 10h4l5-4v12l-5-4H3Zm13-1a5 5 0 0 1 0 6m3-9a9 9 0 0 1 0 12",
    memory: "M8 4h8v4h4v8h-4v4H8v-4H4V8h4Zm2 6h4v4h-4Z",
    close: "m6 6 12 12M6 18 18 6",
    trash: "M5 7h14M9 7V4h6v3m-8 0 1 13h8l1-13",
    mask: "M4 8c4-3 12-3 16 0v5c0 4-4 7-8 7s-8-3-8-7Zm4 3h.01M16 11h.01M9 16c2 1 4 1 6 0",
  };
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] ?? paths.spark} />
    </svg>
  );
}

function errorText(error: unknown) {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "操作未能完成";
}

function blankCharacterCard(): CharacterCard {
  return {
    id: `custom-${crypto.randomUUID()}`,
    name: "新角色",
    tagline: "等待你补完的人物",
    description: "",
    background: "",
    personality: "",
    speech_style: "",
    relationship: "",
    greeting: "你好。",
    scenario: "",
    motivation: "",
    boundaries: "",
    source_notes: "",
    knowledge_boundary: {
      character_kind: "original",
      era: "",
      knowledge_cutoff: "",
      universe: "",
      allowed_knowledge: "",
      forbidden_knowledge: "",
      forbidden_terms: [],
      can_learn_from_user: true,
      allow_everyday_invention: true,
    },
    fixed_facts: [],
    worldbook: [],
    example_dialogue: [],
    generation_preset: {},
    draft_notice: "手动创建角色。请填写并检查全部字段后保存。",
  };
}

function formatTime(value: string) {
  const date = new Date(value);
  return date.toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function messageText(message: Message) {
  const content = message.content.trim();
  if (!message.display) return content;
  // Four-part messages are historical experiments. Flatten them into the
  // natural single-bubble format and intentionally omit inner thoughts/state.
  const speech = (message.display.speech || content).trim();
  const action = message.display.action?.trim();
  if (action && !speech.startsWith("（") && !speech.startsWith("(")) {
    return `（${action}）${speech}`;
  }
  return speech;
}

function CharacterAvatar({
  character,
  size = "normal",
}: {
  character: Character;
  size?: "small" | "normal" | "large";
}) {
  const src = character.avatar_url || PORTRAITS[character.id];
  return (
    <div className={`character-avatar ${size}`}>
      {src ? <img src={src} alt="" /> : <span>{character.name.slice(0, 1)}</span>}
    </div>
  );
}

function Modal({
  title,
  eyebrow,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  eyebrow?: string;
  onClose: () => void;
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <div className="modal-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section className={`modal ${wide ? "wide" : ""}`} role="dialog" aria-modal="true">
        <header>
          <div>
            {eyebrow && <span className="eyebrow">{eyebrow}</span>}
            <h2>{title}</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="关闭">
            <Icon name="close" />
          </button>
        </header>
        {children}
      </section>
    </div>
  );
}

function EmptyConversation({
  character,
  onStarter,
}: {
  character: Character;
  onStarter: (text: string) => void;
}) {
  return (
    <div className="empty-conversation">
      <CharacterAvatar character={character} size="large" />
      <span className="eyebrow">ONLY ONE STORY</span>
      <h2>和{character.name}的故事，从此刻开始</h2>
      <p>{character.description}</p>
      <div className="starter-grid">
        {(character.starters ?? []).map((starter) => (
          <button key={starter} onClick={() => onStarter(starter)}>
            <Icon name="spark" size={18} />
            <span>{starter}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [settings, setSettings] = useState<ProviderSettings | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [preferredProvider, setPreferredProvider] = useState<CloudProviderName>(initialCloudProvider);
  const [activeCharacter, setActiveCharacter] = useState<Character | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [view, setView] = useState<"discover" | "chat">("discover");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [draft, setDraft] = useState("");
  const [streamText, setStreamText] = useState("");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [deleteCharacterTarget, setDeleteCharacterTarget] = useState<Character | null>(null);
  const [editingCard, setEditingCard] = useState<CharacterCard | null>(null);
  const [memory, setMemory] = useState<SessionMemory | null>(null);
  const [memoryNotes, setMemoryNotes] = useState("");
  const [keptMemoryItems, setKeptMemoryItems] = useState<Set<number>>(new Set());
  const [structuredMemories, setStructuredMemories] = useState<StructuredMemoryRecord[]>([]);
  const [structuredMemoryStatus, setStructuredMemoryStatus] = useState<StructuredMemoryStatus | null>(null);
  const [structuredMemoryBusy, setStructuredMemoryBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);

  async function reloadBase() {
    const [nextHealth, nextSettings, nextCharacters] = await Promise.all([
      getHealth(),
      getProviderSettings(),
      getCharacters(),
    ]);
    setHealth(nextHealth);
    setSettings(nextSettings);
    setCharacters(nextCharacters);
    return { nextHealth, nextSettings, nextCharacters };
  }

  useEffect(() => {
    reloadBase()
      .catch((error) => setStatus(errorText(error)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const container = messagesRef.current;
    if (!container) return;
    // Scroll only the message viewport. scrollIntoView can move the page and
    // push the fixed chat header off-screen when a short opening is inserted.
    container.scrollTo({
      top: container.scrollHeight,
      behavior: busy || streamText ? "auto" : "smooth",
    });
  }, [session?.messages.length, streamText, busy]);

  useEffect(() => {
    return () => {
      if (imagePreview) URL.revokeObjectURL(imagePreview);
    };
  }, [imagePreview]);

  const activeCloud = settings?.[preferredProvider];
  const configured = activeCloud?.configured ?? false;
  const usagePercent = useMemo(() => {
    const used = settings?.[preferredProvider].usage.total_tokens ?? 0;
    const limit = settings?.token_limits.daily_token_limit ?? 1;
    return Math.min(100, Math.round((used / limit) * 100));
  }, [preferredProvider, settings]);

  function chooseProvider(provider: CloudProviderName) {
    setPreferredProvider(provider);
    try {
      localStorage.setItem("shiyu-cloud-provider", provider);
    } catch {
      // The selected provider still works for this page session.
    }
  }

  async function openCharacter(character: Character) {
    if (!configured) {
      setShowSettings(true);
      setStatus(`先配置 ${providerLabel(preferredProvider)} API Key，再开始聊天。`);
      return;
    }
    setStatus("正在打开这段故事…");
    try {
      const nextSession = await getRoleConversation(character.id, preferredProvider);
      setActiveCharacter(character);
      setSession(nextSession);
      setView("chat");
      setStatus("");
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  async function refreshSession() {
    if (!session) return;
    const next = await getRoleConversation(session.character_id, preferredProvider);
    setSession(next);
  }

  async function sendMessage(textOverride?: string) {
    if (!session || !activeCharacter || busy) return;
    const text = (textOverride ?? draft).trim();
    if (!text) return;
    setBusy(true);
    setStatus("");
    setStreamText("");
    setDraft("");
    const controller = new AbortController();
    abortRef.current = controller;
    let imageId: string | undefined;
    const optimistic: Message = {
      id: -Date.now(),
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setSession((current) => current ? {
      ...current,
      messages: [...current.messages, optimistic],
    } : current);
    try {
      if (selectedImage) {
        setStatus("正在准备图片…");
        const uploaded = await uploadImage(session.id, selectedImage);
        imageId = uploaded.id;
      }
      setStatus("正在回应…");
      await streamChat(
        {
          session_id: session.id,
          message: text,
          image_id: imageId,
          max_new_tokens: 768,
          request_id: crypto.randomUUID(),
          timeout_seconds: 180,
        },
        (event) => {
          if (event.event === "delta") {
            setStreamText((current) => current + String(event.data.text ?? ""));
          } else if (event.event === "done") {
            setStatus("");
          } else if (event.event === "error") {
            setStatus(String(event.data.detail ?? "生成失败"));
          }
        },
        controller.signal,
      );
      await refreshSession();
      setStreamText("");
      clearImage();
      await reloadBase();
    } catch (error) {
      if (controller.signal.aborted) setStatus("已停止生成，本轮未保存。");
      else setStatus(errorText(error));
      await refreshSession().catch(() => undefined);
      setStreamText("");
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  async function stop() {
    if (!session) return;
    abortRef.current?.abort();
    await stopGeneration(session.id).catch(() => undefined);
  }

  function chooseImage(file: File | undefined) {
    if (!file) return;
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));
  }

  function clearImage() {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setSelectedImage(null);
    setImagePreview("");
    if (fileRef.current) fileRef.current.value = "";
  }

  async function openMemory() {
    if (!session || !activeCharacter) return;
    try {
      const [loaded, structured, structuredStatus] = await Promise.all([
        getMemory(session.id),
        health?.structured_memory_enabled
          ? getStructuredMemories(activeCharacter.id)
          : Promise.resolve(null),
        health?.structured_memory_enabled
          ? getStructuredMemoryStatus(activeCharacter.id)
          : Promise.resolve(null),
      ]);
      setMemory(loaded);
      setMemoryNotes(loaded.notes);
      setKeptMemoryItems(new Set(loaded.items.map((_, index) => index)));
      setStructuredMemories(structured?.records ?? []);
      setStructuredMemoryStatus(structuredStatus);
      setShowMemory(true);
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  async function refreshStructuredMemories() {
    if (!activeCharacter || !health?.structured_memory_enabled) return;
    const [loaded, nextStatus] = await Promise.all([
      getStructuredMemories(activeCharacter.id),
      getStructuredMemoryStatus(activeCharacter.id),
    ]);
    setStructuredMemories(loaded.records);
    setStructuredMemoryStatus(nextStatus);
  }

  async function deleteMemoryRecord(record: StructuredMemoryRecord) {
    if (!activeCharacter || structuredMemoryBusy) return;
    setStructuredMemoryBusy(true);
    try {
      await reviewStructuredMemory(activeCharacter.id, {
        action: "delete",
        item_id: record.id,
      });
      await refreshStructuredMemories();
      setStatus("记忆已删除，系统不会用同一段旧消息重新创建它");
    } catch (error) {
      setStatus(errorText(error));
    } finally {
      setStructuredMemoryBusy(false);
    }
  }

  async function openCharacterEditor() {
    if (!activeCharacter?.custom) {
      setStatus("内置角色档案暂不允许修改。");
      return;
    }
    try {
      const card = await getCharacterCard(activeCharacter.id);
      setEditingCard(card);
      setShowCreate(true);
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  async function confirmDeleteCharacter() {
    const target = deleteCharacterTarget;
    if (!target || busy) return;
    setBusy(true);
    setStatus(`正在删除${target.name}…`);
    try {
      await deleteCharacter(target.id);
      const refreshed = await reloadBase();
      if (activeCharacter?.id === target.id) {
        setActiveCharacter(null);
        setSession(null);
        setMemory(null);
        setStructuredMemories([]);
        setStructuredMemoryStatus(null);
        setDraft("");
        setStreamText("");
        clearImage();
        setView("discover");
      }
      setDeleteCharacterTarget(null);
      setEditingCard(null);
      setShowCreate(false);
      setShowMemory(false);
      setStatus(`${target.name}已删除，相关聊天、记忆和素材已清理`);
      if (!refreshed.nextCharacters.some((item) => item.id === target.id)) {
        setCharacters(refreshed.nextCharacters);
      }
    } catch (error) {
      setStatus(errorText(error));
    } finally {
      setBusy(false);
    }
  }

  async function persistMemory() {
    if (!session || !memory) return;
    try {
      const saved = await saveMemory(session.id, {
        notes: memoryNotes,
        auto_enabled: memory.auto_enabled,
        revision: memory.revision,
        kept_items: [...keptMemoryItems].sort((left, right) => left - right),
      });
      setMemory(saved);
      setKeptMemoryItems(new Set(saved.items.map((_, index) => index)));
      setStatus("记忆已保存");
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  async function resetConversation() {
    if (!activeCharacter || !window.confirm(
      `确定重置与${activeCharacter.name}的全部对话、长期记忆、图片和角色状态吗？`,
    )) return;
    try {
      const next = await resetRoleConversation(activeCharacter.id);
      setSession(next);
      setStreamText("");
      setStatus("角色对话已重置");
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  async function playMessage(message: Message) {
    if (!session || message.role !== "assistant" || message.id < 0) return;
    try {
      setStatus("正在合成语音…");
      const url = await synthesize(session.id, message.id);
      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      await audio.play();
      setStatus("");
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  async function toggleRecording() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size) audioChunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
        const blob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        try {
          setStatus("正在识别语音…");
          const text = await transcribe(blob);
          setDraft((current) => current + text);
          setStatus("语音已转成文字，请确认后发送");
        } catch (error) {
          setStatus(errorText(error));
        }
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  async function generatedCharacter(card: CharacterCard) {
    const refreshed = await reloadBase();
    setShowCreate(false);
    const next = refreshed.nextCharacters.find((item) => item.id === card.id);
    if (next) await openCharacter(next);
  }

  if (loading) {
    return (
      <div className="boot-screen">
        <div className="brand-mark">拾</div>
        <p>正在翻开你的故事…</p>
      </div>
    );
  }

  const messages = session?.messages ?? [];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setView("discover")}>
          <span className="brand-mark">拾</span>
          <span><strong>拾语</strong><small>SHIYU</small></span>
        </button>
        <p className="side-intro">每一次对话，只属于一个角色。</p>
        <nav>
          <button className={view === "discover" ? "active" : ""} onClick={() => setView("discover")}>
            <Icon name="spark" /> 发现角色
          </button>
          <button onClick={() => {
            setEditingCard(null);
            setShowCreate(true);
          }}>
            <Icon name="plus" /> 创建角色
          </button>
          <button onClick={() => setShowSettings(true)}>
            <Icon name="settings" /> 应用设置
          </button>
        </nav>
        <div className="sidebar-characters">
          <span>角色</span>
          {characters.slice(0, 6).map((character) => (
            <button
              key={character.id}
              className={activeCharacter?.id === character.id ? "selected" : ""}
              onClick={() => openCharacter(character)}
            >
              <CharacterAvatar character={character} size="small" />
              <span><strong>{character.name}</strong><small>{character.tag}</small></span>
            </button>
          ))}
        </div>
        <div className="provider-card">
          <span className={configured ? "online" : ""} />
          <div>
            <strong>{configured ? `${providerLabel(preferredProvider)} 已连接` : "尚未配置 API"}</strong>
            <small>{activeCloud?.model_id ?? (preferredProvider === "opencode" ? "deepseek-v4.1-flash" : "deepseek-flash")}</small>
          </div>
        </div>
      </aside>

      <main className="workspace">
        {view === "discover" ? (
          <section className="discover">
            <header className="page-top">
              <span>发现</span>
              <div>
                <span className="usage-pill">今日 {activeCloud?.usage.total_tokens ?? 0} tokens</span>
                <button className="icon-button" onClick={() => setShowSettings(true)}>
                  <Icon name="settings" />
                </button>
              </div>
            </header>
            <div className="hero">
              <span className="eyebrow">A STORY THAT REMEMBERS</span>
              <h1>不只回复一句话，<br /><em>而是记住这段故事。</em></h1>
              <p>每位角色只有一个持续对话。你说过的、她经历过的，都会留在这段关系中。</p>
              <button className="primary-button" onClick={() => {
                setEditingCard(null);
                setShowCreate(true);
              }}>
                <Icon name="plus" /> 创建一位角色
              </button>
            </div>
            <div className="section-heading">
              <div><span className="eyebrow">CHARACTERS</span><h2>此刻，想和谁聊聊？</h2></div>
              <span>{characters.length} 位角色</span>
            </div>
            <div className="character-grid">
              {characters.map((character, index) => (
                <div key={character.id} className="character-card-shell">
                  <button
                    className={`character-card card-${index % 4}`}
                    onClick={() => openCharacter(character)}
                  >
                    <div className="character-art">
                      {character.avatar_url || PORTRAITS[character.id]
                        ? <img src={character.avatar_url || PORTRAITS[character.id]} alt="" />
                        : <div className="custom-art"><Icon name="mask" size={58} /></div>}
                      <span>{character.tag}</span>
                      <div className="card-gradient" />
                      <div className="card-copy">
                        <small>{character.profile?.scene || "原创角色"}</small>
                        <h3>{character.name}</h3>
                        <p>{character.profile?.line || character.description}</p>
                      </div>
                    </div>
                    <div className="card-footer">
                      <span>{character.profile?.tags?.join(" · ") || "原创 · AI 角色"}</span>
                      <b>进入故事</b>
                    </div>
                  </button>
                  {character.custom && (
                    <button
                      className="character-delete"
                      title={`删除${character.name}`}
                      aria-label={`删除${character.name}`}
                      disabled={busy}
                      onClick={() => setDeleteCharacterTarget(character)}
                    >
                      <Icon name="trash" size={16} />
                    </button>
                  )}
                </div>
              ))}
            </div>
            <footer className="page-footer">
              AI 的回答可能不准确 · 对话与记忆默认保存在本机 · 图片和文字会发送给 {providerLabel(preferredProvider)}
            </footer>
          </section>
        ) : activeCharacter && session ? (
          <section className="chat-screen">
            {activeCharacter.avatar_url && (
              <div
                className="chat-background"
                style={{ backgroundImage: `url("${activeCharacter.avatar_url}")` }}
              />
            )}
            <header className="chat-top">
              <button className="back-button" onClick={() => setView("discover")}>←</button>
              <CharacterAvatar character={activeCharacter} size="small" />
              <div className="chat-title">
                <h1>{activeCharacter.name}</h1>
                <p>{activeCharacter.tag} · 唯一对话</p>
              </div>
              <div className="chat-actions">
                <button onClick={openMemory}><Icon name="memory" /> 记忆</button>
                <button onClick={openCharacterEditor}><Icon name="mask" /> 档案</button>
                <button className="danger-text" onClick={resetConversation}><Icon name="trash" /></button>
              </div>
            </header>
            <div className="messages" ref={messagesRef}>
              {messages.length === 0 && (
                <EmptyConversation character={activeCharacter} onStarter={(text) => setDraft(text)} />
              )}
              {messages.map((message) => (
                <article key={message.id} className={`message ${message.role}`}>
                  {message.role === "assistant" && <CharacterAvatar character={activeCharacter} size="small" />}
                  <div className="message-body">
                    <div className="bubble">
                      {messageText(message).split("\n").map((line, index) => (
                        <span key={index}>{line}<br /></span>
                      ))}
                      {message.image && (
                        <img
                          className="message-image"
                          src={`/images/${message.image.id}?session_id=${session.id}`}
                          alt="用户上传的图片"
                        />
                      )}
                    </div>
                    <div className="message-meta">
                      <span>{formatTime(message.created_at)}</span>
                      {message.role === "assistant" && (
                        <button onClick={() => playMessage(message)} aria-label="朗读">
                          <Icon name="volume" size={16} /> 朗读
                        </button>
                      )}
                    </div>
                  </div>
                </article>
              ))}
              {busy && (
                <article className="message assistant streaming">
                  <CharacterAvatar character={activeCharacter} size="small" />
                  <div className="message-body">
                    <div className="bubble">
                      {streamText || <span className="typing"><i /><i /><i /></span>}
                    </div>
                  </div>
                </article>
              )}
            </div>
            <div className="composer-wrap">
              {imagePreview && (
                <div className="image-preview">
                  <img src={imagePreview} alt="" />
                  <button onClick={clearImage}><Icon name="close" size={16} /></button>
                </div>
              )}
              {status && <p className="status-line">{status}</p>}
              <div className="composer">
                <textarea
                  value={draft}
                  maxLength={4096}
                  rows={2}
                  placeholder={`和${activeCharacter.name}说点什么…`}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      sendMessage();
                    }
                  }}
                />
                <div className="composer-actions">
                  <div>
                    <input
                      ref={fileRef}
                      type="file"
                      accept="image/jpeg,image/png,image/gif,image/webp"
                      hidden
                      onChange={(event) => chooseImage(event.target.files?.[0])}
                    />
                    <button className="icon-button" onClick={() => fileRef.current?.click()} title="上传图片">
                      <Icon name="image" />
                    </button>
                    <button
                      className={`icon-button ${recording ? "recording" : ""}`}
                      onClick={toggleRecording}
                      title="语音输入"
                    >
                      <Icon name="mic" />
                    </button>
                  </div>
                  {busy ? (
                    <button className="stop-button" onClick={stop}><Icon name="stop" /> 停止</button>
                  ) : (
                    <button className="send-button" disabled={!draft.trim()} onClick={() => sendMessage()}>
                      发送 <Icon name="send" />
                    </button>
                  )}
                </div>
              </div>
              <p className="composer-note">
                Enter 发送 · Shift + Enter 换行 · 图片支持由 {providerLabel(preferredProvider)} 提供
              </p>
            </div>
          </section>
        ) : null}
      </main>

      {showSettings && (
        <SettingsModal
          settings={settings}
          provider={preferredProvider}
          onProviderChange={chooseProvider}
          onClose={() => setShowSettings(false)}
          onSaved={async () => {
            await reloadBase();
            setStatus("设置已保存");
          }}
          usagePercent={usagePercent}
        />
      )}
      {showMemory && activeCharacter && session && memory && (
        <Modal title={`${activeCharacter.name}的记忆`} eyebrow="LONG-TERM MEMORY" onClose={() => setShowMemory(false)}>
          <div className="memory-summary">
            <strong>{memory.items.length}</strong>
            <span>条来自早期对话的原文记忆</span>
          </div>
          <label className="field">
            <span>用户修订</span>
            <textarea value={memoryNotes} onChange={(event) => setMemoryNotes(event.target.value)} rows={4} />
          </label>
          <label className="memory-auto">
            <input
              type="checkbox"
              checked={memory.auto_enabled}
              onChange={(event) => setMemory((current) => current ? {
                ...current,
                auto_enabled: event.target.checked,
              } : current)}
            />
            <span>允许在长对话中自动整理早期记忆</span>
          </label>
          <div className="memory-list">
            {memory.items.length === 0 && <p className="empty-copy">长对话产生早期摘要后，这里会显示可追溯的原文记忆。</p>}
            {memory.items.map((item, index) => (
              <label key={`${item.message_id}-${item.quote}`}>
                <input
                  type="checkbox"
                  checked={keptMemoryItems.has(index)}
                  onChange={(event) => {
                    setKeptMemoryItems((current) => {
                      const next = new Set(current);
                      if (event.target.checked) next.add(index);
                      else next.delete(index);
                      return next;
                    });
                  }}
                />
                <span>{item.role === "user" ? "用户" : activeCharacter.name}</span>
                <p>{item.quote}</p>
              </label>
            ))}
          </div>
          {health?.structured_memory_enabled && (
            <section className="structured-memory-section">
              <div className="setting-heading">
                <div>
                  <h3>结构化长期记忆</h3>
                  <p>聊天过程中由后台自动整理，不需要逐条确认。你可以删除或直接在对话中更正；删除一项事实会同时移除它的历史版本。</p>
                  {structuredMemoryStatus && (
                    <p className="memory-status">
                      {(structuredMemoryStatus.jobs.failed ?? 0) > 0
                        ? "部分记忆整理暂时失败，聊天记录和已有记忆不受影响"
                        : (structuredMemoryStatus.jobs.running ?? 0) > 0
                        ? "正在整理最近的对话"
                        : (structuredMemoryStatus.jobs.queued ?? 0) > 0
                          ? "最近的对话等待后台整理"
                          : "后台记忆已整理到最新"}
                      {structuredMemoryStatus.usage_today.calls > 0
                        ? ` · 今日 ${structuredMemoryStatus.usage_today.total_tokens.toLocaleString()} / ${structuredMemoryStatus.daily_token_limit.toLocaleString()} Token`
                        : ""}
                    </p>
                  )}
                </div>
                <button className="secondary-button" disabled={structuredMemoryBusy} onClick={refreshStructuredMemories}>
                  {structuredMemoryBusy ? "读取中…" : "刷新"}
                </button>
              </div>
              <div className="memory-list structured-memory-list">
                {structuredMemories.length === 0 && (
                  <p className="empty-copy">新一轮聊天完成后，系统会在后台自动整理值得长期保留的信息。</p>
                )}
                {structuredMemories.map((record) => {
                  return (
                    <article className={`structured-memory-record ${record.state}`} key={record.id}>
                      <div>
                        <span>
                          {record.state === "active"
                            ? "当前"
                            : record.state === "contested"
                              ? "有疑义"
                              : record.state === "superseded"
                                ? "历史旧值"
                                : record.state === "candidate"
                                  ? "历史草稿"
                                  : "已删除"}
                        </span>
                        <strong>{record.content}</strong>
                      </div>
                      <p>
                        {record.value ? `值：${record.value} · ` : ""}
                        来源消息：{record.evidence.map((item) => item.message_id).join("、")}
                      </p>
                      <div className="structured-memory-actions">
                        {record.state !== "deleted" && (
                          <button
                            className="danger-text"
                            disabled={structuredMemoryBusy}
                            onClick={() => deleteMemoryRecord(record)}
                          >
                            删除
                          </button>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          )}
          <div className="modal-actions">
            <button className="secondary-button" onClick={() => setShowMemory(false)}>关闭</button>
            <button className="primary-button" onClick={persistMemory}>保存修订</button>
          </div>
        </Modal>
      )}
      {showCreate && (
        <CreateCharacterModal
          initialCard={editingCard}
          provider={preferredProvider}
          onClose={() => {
            setShowCreate(false);
            setEditingCard(null);
          }}
          onSaved={generatedCharacter}
        />
      )}
      {deleteCharacterTarget && (
        <Modal
          title={`删除${deleteCharacterTarget.name}？`}
          eyebrow="DELETE CHARACTER"
          onClose={() => !busy && setDeleteCharacterTarget(null)}
        >
          <p>
            这会永久删除该角色的聊天、长期记忆、内部状态、头像记录和 App
            生成素材。手机相册中的原图不会被删除，此操作无法撤销。
          </p>
          <div className="modal-actions">
            <button className="secondary-button" disabled={busy} onClick={() => setDeleteCharacterTarget(null)}>
              取消
            </button>
            <button className="stop-button" disabled={busy} onClick={confirmDeleteCharacter}>
              {busy ? "正在删除…" : "永久删除"}
            </button>
          </div>
        </Modal>
      )}
      {!configured && !showSettings && (
        <div className="setup-banner">
          <span>使用 {providerLabel(preferredProvider)} 前需要填写自己的 API Key</span>
          <button onClick={() => setShowSettings(true)}>现在配置</button>
        </div>
      )}
    </div>
  );
}

function SettingsModal({
  settings,
  provider,
  onProviderChange,
  onClose,
  onSaved,
  usagePercent,
}: {
  settings: ProviderSettings | null;
  provider: CloudProviderName;
  onProviderChange: (provider: CloudProviderName) => void;
  onClose: () => void;
  onSaved: () => Promise<void>;
  usagePercent: number;
}) {
  const [openCodeKey, setOpenCodeKey] = useState("");
  const [deepSeekKey, setDeepSeekKey] = useState("");
  const [requestLimit, setRequestLimit] = useState(settings?.token_limits.request_token_limit ?? 12000);
  const [dailyLimit, setDailyLimit] = useState(settings?.token_limits.daily_token_limit ?? 200000);
  const [message, setMessage] = useState("");
  const active = settings?.[provider];

  async function save() {
    try {
      if (openCodeKey.trim()) await saveOpenCodeKey(openCodeKey.trim());
      if (deepSeekKey.trim()) await saveDeepSeekKey(deepSeekKey.trim());
      await saveTokenLimits(requestLimit, dailyLimit);
      setOpenCodeKey("");
      setDeepSeekKey("");
      setMessage("设置已保存");
      await onSaved();
    } catch (error) {
      setMessage(errorText(error));
    }
  }

  async function removeKey(target: CloudProviderName) {
    if (!window.confirm(`从这台电脑删除 ${providerLabel(target)} API Key？`)) return;
    try {
      if (target === "opencode") await deleteOpenCodeKey();
      else await deleteDeepSeekKey();
      setMessage(`${providerLabel(target)} API Key 已删除`);
      await onSaved();
    } catch (error) {
      setMessage(errorText(error));
    }
  }

  return (
    <Modal title="应用设置" eyebrow="MODEL & LIMITS" onClose={onClose}>
      <section className="setting-section">
        <div className="setting-heading">
          <div>
            <h3>当前聊天 Provider</h3>
            <p>切换后，各角色继续使用原聊天记录和长期记忆。</p>
          </div>
          <span className={`provider-state ${active?.configured ? "ready" : ""}`}>
            {active?.configured ? "已配置" : "未配置"}
          </span>
        </div>
        <div className="provider-choice" role="group" aria-label="聊天 Provider">
          <button
            className={provider === "opencode" ? "active" : ""}
            onClick={() => onProviderChange("opencode")}
          >
            OpenCode Go
          </button>
          <button
            className={provider === "deepseek" ? "active" : ""}
            onClick={() => onProviderChange("deepseek")}
          >
            DeepSeek
          </button>
        </div>
      </section>
      <section className="setting-section">
        <div className="setting-heading">
          <div>
            <h3>OpenCode Go API</h3>
            <p>Key 只在 FastAPI 后端使用，加密保存在本机。</p>
          </div>
          <span className={`provider-state ${settings?.opencode.configured ? "ready" : ""}`}>
            {settings?.opencode.configured ? "已配置" : "未配置"}
          </span>
        </div>
        <label className="field">
          <span>{settings?.opencode.configured ? "替换 API Key（留空则不变）" : "OpenCode Go API Key"}</span>
          <input
            type="password"
            value={openCodeKey}
            placeholder="oc_sk_..."
            autoComplete="off"
            onChange={(event) => setOpenCodeKey(event.target.value)}
          />
        </label>
        <div className="provider-facts">
          <span>模型 <strong>{settings?.opencode.model_id ?? "deepseek-v4.1-flash"}</strong></span>
          <span>支持文字、图片</span>
          <span>OpenAI 兼容接口</span>
        </div>
        {settings?.opencode.configured && (
          <button className="danger-text" onClick={() => removeKey("opencode")}>删除 OpenCode Key</button>
        )}
      </section>
      <section className="setting-section">
        <div className="setting-heading">
          <div>
            <h3>DeepSeek 备用接口</h3>
            <p>保留原有调用路径，可在上方切换后使用。</p>
          </div>
          <span className={`provider-state ${settings?.deepseek.configured ? "ready" : ""}`}>
            {settings?.deepseek.configured ? "已配置" : "未配置"}
          </span>
        </div>
        <label className="field">
          <span>{settings?.deepseek.configured ? "替换 API Key（留空则不变）" : "DeepSeek API Key"}</span>
          <input
            type="password"
            value={deepSeekKey}
            placeholder="sk-..."
            autoComplete="off"
            onChange={(event) => setDeepSeekKey(event.target.value)}
          />
        </label>
        <div className="provider-facts">
          <span>模型 <strong>{settings?.deepseek.model_id ?? "deepseek-flash"}</strong></span>
          <span>支持文字、图片</span>
        </div>
        {settings?.deepseek.configured && (
          <button className="danger-text" onClick={() => removeKey("deepseek")}>删除 DeepSeek Key</button>
        )}
      </section>
      <section className="setting-section">
        <div className="setting-heading">
          <div><h3>Token 限额</h3><p>防止误操作产生过高费用。</p></div>
        </div>
        <div className="two-fields">
          <label className="field">
            <span>单次请求上限</span>
            <input
              type="number"
              min={256}
              value={requestLimit}
              onChange={(event) => setRequestLimit(Number(event.target.value))}
            />
          </label>
          <label className="field">
            <span>每日上限</span>
            <input
              type="number"
              min={1000}
              value={dailyLimit}
              onChange={(event) => setDailyLimit(Number(event.target.value))}
            />
          </label>
        </div>
        <div className="usage-line">
          <span>{providerLabel(provider)} 今日已用 {active?.usage.total_tokens ?? 0} tokens</span>
          <div><i style={{ width: `${usagePercent}%` }} /></div>
        </div>
      </section>
      <section className="setting-section compact">
        <h3>本地功能</h3>
        <p>网页语音使用现有的 Whisper 与 GPT-SoVITS。本地 Qwen 入口暂时隐藏。</p>
      </section>
      {message && <p className="form-message">{message}</p>}
      <div className="modal-actions">
        <span className="spacer" />
        <button className="secondary-button" onClick={onClose}>取消</button>
        <button className="primary-button" onClick={save}>保存设置</button>
      </div>
    </Modal>
  );
}

function CreateCharacterModal({
  initialCard,
  provider,
  onClose,
  onSaved,
}: {
  initialCard?: CharacterCard | null;
  provider: CloudProviderName;
  onClose: () => void;
  onSaved: (card: CharacterCard) => Promise<void>;
}) {
  const [inspiration, setInspiration] = useState("");
  const [card, setCard] = useState<CharacterCard | null>(initialCard ?? null);
  const [rawCard, setRawCard] = useState(initialCard ? JSON.stringify(initialCard, null, 2) : "");
  const [status, setStatus] = useState("");
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    if (card) setRawCard(JSON.stringify(card, null, 2));
  }, [card]);

  async function generate() {
    if (!inspiration.trim()) return;
    setGenerating(true);
    setStatus(`${providerLabel(provider)} 正在整理角色档案…`);
    try {
      const response = await fetch("/characters/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inspiration, provider }),
      });
      const body = await response.json();
      if (!response.ok) throw new ApiError(response.status, body.detail ?? body);
      setCard(body.card);
      setStatus("草稿已生成，请检查认知边界和作者事实。");
    } catch (error) {
      setStatus(errorText(error));
    } finally {
      setGenerating(false);
    }
  }

  function patch<K extends keyof CharacterCard>(key: K, value: CharacterCard[K]) {
    setCard((current) => current ? { ...current, [key]: value } : current);
  }

  function patchBoundary<K extends keyof CharacterCard["knowledge_boundary"]>(
    key: K,
    value: CharacterCard["knowledge_boundary"][K],
  ) {
    if (!card) return;
    setCard({ ...card, knowledge_boundary: { ...card.knowledge_boundary, [key]: value } });
  }

  async function save() {
    if (!card) return;
    setStatus("正在保存角色…");
    try {
      const response = await fetch("/characters", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ card }),
      });
      const body = await response.json();
      if (!response.ok) throw new ApiError(response.status, body.detail ?? body);
      await onSaved(body.card);
    } catch (error) {
      setStatus(errorText(error));
    }
  }

  if (!card) {
    return (
      <Modal title="创建一位角色" eyebrow="CHARACTER STUDIO" onClose={onClose}>
        <p className="modal-lead">
          写下身份、时代、关系和说话方式。生成内容只是草稿，正式事实需要你确认。
        </p>
        <label className="field">
          <span>角色灵感</span>
          <textarea
            rows={5}
            maxLength={1800}
            value={inspiration}
            placeholder="例如：李白身处唐代；以诗酒和游历为生活重心；说话豪放、敏锐；不能知道现代科技和后世历史。"
            onChange={(event) => setInspiration(event.target.value)}
          />
        </label>
        {status && <p className="form-message">{status}</p>}
        <div className="modal-actions">
          <button className="secondary-button" onClick={() => setCard(blankCharacterCard())}>
            手动创建
          </button>
          <button className="secondary-button" onClick={onClose}>取消</button>
          <button className="primary-button" disabled={generating || !inspiration.trim()} onClick={generate}>
            {generating ? "正在生成…" : "生成角色草稿"}
          </button>
        </div>
      </Modal>
    );
  }

  return (
    <Modal title="检查角色档案" eyebrow="REVIEW BEFORE SAVING" onClose={onClose} wide>
      <div className="draft-notice">{card.draft_notice || "请检查所有正式事实后再保存。"}</div>
      <div className="form-grid">
        <label className="field"><span>名字</span><input value={card.name} onChange={(e) => patch("name", e.target.value)} /></label>
        <label className="field"><span>一句话介绍</span><input value={card.tagline} onChange={(e) => patch("tagline", e.target.value)} /></label>
        <label className="field full"><span>简介</span><textarea rows={2} value={card.description} onChange={(e) => patch("description", e.target.value)} /></label>
        <label className="field full"><span>背景</span><textarea rows={3} value={card.background} onChange={(e) => patch("background", e.target.value)} /></label>
        <label className="field"><span>性格</span><textarea rows={3} value={card.personality} onChange={(e) => patch("personality", e.target.value)} /></label>
        <label className="field"><span>说话方式</span><textarea rows={3} value={card.speech_style} onChange={(e) => patch("speech_style", e.target.value)} /></label>
        <label className="field"><span>与用户的关系</span><textarea rows={2} value={card.relationship} onChange={(e) => patch("relationship", e.target.value)} /></label>
        <label className="field"><span>开场白</span><textarea rows={2} value={card.greeting} onChange={(e) => patch("greeting", e.target.value)} /></label>
      </div>
      <h3 className="form-section-title">认知边界</h3>
      <div className="form-grid">
        <label className="field">
          <span>角色类型</span>
          <select
            value={card.knowledge_boundary.character_kind}
            onChange={(e) => patchBoundary("character_kind", e.target.value as CharacterCard["knowledge_boundary"]["character_kind"])}
          >
            <option value="original">原创角色</option>
            <option value="historical">真实历史人物</option>
            <option value="fictional_canon">已有虚构作品角色</option>
          </select>
        </label>
        <label className="field"><span>时代 / 世界</span><input value={card.knowledge_boundary.era} onChange={(e) => patchBoundary("era", e.target.value)} /></label>
        <label className="field"><span>知识截止</span><input value={card.knowledge_boundary.knowledge_cutoff} onChange={(e) => patchBoundary("knowledge_cutoff", e.target.value)} /></label>
        <label className="field"><span>作品 / 世界范围</span><input value={card.knowledge_boundary.universe} onChange={(e) => patchBoundary("universe", e.target.value)} /></label>
        <label className="field full"><span>允许知道的内容</span><textarea rows={3} value={card.knowledge_boundary.allowed_knowledge} onChange={(e) => patchBoundary("allowed_knowledge", e.target.value)} /></label>
        <label className="field full"><span>禁止知道的内容</span><textarea rows={3} value={card.knowledge_boundary.forbidden_knowledge} onChange={(e) => patchBoundary("forbidden_knowledge", e.target.value)} /></label>
        <label className="field full">
          <span>禁止知识关键词</span>
          <input
            value={card.knowledge_boundary.forbidden_terms.join("，")}
            placeholder="Python，人工智能，智能手机"
            onChange={(e) => patchBoundary("forbidden_terms", e.target.value.split(/[，,\n]/).map((v) => v.trim()).filter(Boolean))}
          />
        </label>
        <label className="check-field">
          <input type="checkbox" checked={card.knowledge_boundary.can_learn_from_user} onChange={(e) => patchBoundary("can_learn_from_user", e.target.checked)} />
          <span>用户解释后，角色可以作为“听用户说的”理解</span>
        </label>
        <label className="check-field">
          <input type="checkbox" checked={card.knowledge_boundary.allow_everyday_invention} onChange={(e) => patchBoundary("allow_everyday_invention", e.target.checked)} />
          <span>允许角色虚构普通日常细节</span>
        </label>
      </div>
      <label className="field">
        <span>完整 JSON（高级编辑）</span>
        <textarea
          className="code-area"
          rows={10}
          value={rawCard}
          onChange={(event) => {
            setRawCard(event.target.value);
          }}
          onBlur={() => {
            try {
              setCard(JSON.parse(rawCard));
              setStatus("JSON 格式有效");
            } catch {
              setStatus("JSON 格式暂时无效");
            }
          }}
        />
      </label>
      {status && <p className="form-message">{status}</p>}
      <div className="modal-actions">
        <button className="secondary-button" onClick={onClose}>取消</button>
        <button className="primary-button" onClick={save}>保存角色</button>
      </div>
    </Modal>
  );
}
