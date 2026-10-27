const $ = (id) => document.getElementById(id);
const paths = {
  profile: "M12 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8M4 21v-2a8 8 0 0 1 16 0v2",
  compass: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18m-3 6 7-1-2 7-6 1Z",
  chat: "M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 3V6a2 2 0 0 1 2-2Zm2 5h10M7 13h7",
  settings:
    "m9 3-1 3-3 1-2 3 2 2v3l3 2 1 3h5l1-3 3-2v-3l2-2-2-3-3-1-1-3ZM9 11a3 3 0 1 0 6 0 3 3 0 0 0-6 0",
  sliders: "M4 7h16M4 17h16M8 4v6m8 4v6",
  arrow: "M4 12h15m-6-6 6 6-6 6",
  plus: "M12 5v14M5 12h14",
  close: "m6 6 12 12M6 18 18 6",
  refresh: "M20 7v5h-5M4 17v-5h5M6 6a8 8 0 0 1 14 6M4 12a8 8 0 0 0 14 6",
  mic: "M9 5a3 3 0 0 1 6 0v6a3 3 0 0 1-6 0Zm-3 5v1a6 6 0 0 0 12 0v-1M12 17v4m-3 0h6",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
  upload: "M12 16V4m-5 5 5-5 5 5M4 17v4h16v-4",
  search: "M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14m5 12 6 6",
  sparkles: "m12 3 2 6 6 3-6 2-2 7-2-7-7-2 7-3Zm7 0v4m-2-2h4",
  leaf: "M5 18C2 8 9 3 21 3c0 13-5 19-13 16M5 21 16 10",
  copy: "M8 8h12v13H8ZM4 16V3h12",
  volume: "m3 9 5 0 5-5v16l-5-5H3ZM17 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14",
};
const icon = (name) =>
  `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name] || paths.chat}"/></svg>`;
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
function icons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = icon(el.dataset.icon);
  });
}
const state = {
  characters: [],
  sessions: [],
  historySessions: [],
  historyOffset: 0,
  historyHasMore: false,
  historyLoading: false,
  historyVersion: 0,
  current: null,
  health: {},
  busy: null,
  draftVersion: 0,
  voiceFile: null,
  recording: null,
  recordPending: false,
  recordStream: null,
  recordTimer: null,
  voiceEpoch: 0,
  audioURL: null,
  installPrompt: null,
  pendingDeleteSessionId: null,
  voiceProfileCharacterId: null,
  autoScroll: true,
  draftStorageWarning: false,
};
const storage = {
  get(key, fallback) {
    try {
      return localStorage.getItem("shiyu." + key) || fallback;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem("shiyu." + key, value);
      return true;
    } catch {
      return false;
    }
  },
  remove(key) {
    try {
      localStorage.removeItem("shiyu." + key);
    } catch {}
  },
};
function saveDraft(id = state.current?.id, text = $("draft").value) {
  if (!id) return;
  if (!storage.set("draft." + id, text) && !state.draftStorageWarning) {
    state.draftStorageWarning = true;
    toast("浏览器暂时无法保存草稿，离开或刷新前请先复制输入内容。");
  }
}
let toastTimer;
function toast(text) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    $("toast").hidden = true;
  }, 5000);
}
async function withResponse(path, options, consume, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      ...options,
      signal: controller.signal,
    });
    return await consume(response);
  } catch (error) {
    if (controller.signal.aborted)
      throw new Error(
        "等待服务超时，请检查连接后重试；操作可能已完成，请先核对历史。",
      );
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
async function request(path, options = {}, timeoutMs = 15000) {
  return withResponse(
    path,
    options,
    async (response) => {
      if (!response.ok) {
        let detail;
        try {
          detail = (await response.json()).detail;
        } catch {
          detail = "请求未完成，请稍后重试。";
        }
        throw new Error(
          typeof detail === "string"
            ? detail
            : `请求失败（${response.status}）`,
        );
      }
      return response.json();
    },
    timeoutMs,
  );
}
const jsonPost = (path, body, timeoutMs = 15000) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, timeoutMs);
function character(id) {
  return state.characters.find((c) => c.id === id) || state.characters[0];
}
function setBusy(kind) {
  state.busy = kind;
  const readOnly = !!(
    state.current?.model_id && state.current.model_id !== state.health.model_id
  );
  $("send-message").hidden = kind === "chat";
  $("stop-message").hidden = kind !== "chat";
  $("send-message").disabled = !!kind || readOnly;
  $("open-voice").disabled = !!kind || readOnly || !state.health.asr_provider;
  $("retry-message").disabled = !!kind || readOnly;
  $("draft").readOnly = readOnly;
  $("draft").placeholder = readOnly
    ? "这是旧模型的历史，点击上方“新对话”继续。"
    : "说点什么吧…";
  $("transcribe-audio").disabled = !!kind || !state.voiceFile;
  document
    .querySelectorAll("[data-character]")
    .forEach((b) => (b.disabled = !!kind));
}
async function refreshHealth() {
  try {
    const health = await withResponse(
      "/health",
      {},
      async (response) => {
        if (!response.ok && response.status !== 503)
          throw new Error("服务状态读取失败");
        return response.json();
      },
      5000,
    );
    state.health = health;
    $("connection-label").textContent =
      health.status === "ready" ? "本机服务已连接" : "模型未就绪，可查看历史";
    $("connection-dot").style.background =
      health.status === "ready" ? "#8ba982" : "#c3967e";
    $("model-info").textContent = health.model_id
      ? `当前：${health.model_id}。未启用云端模型。`
      : "模型未加载，历史仍可读取。";
    $("model-caption").textContent = (
      state.current?.model_id ||
      health.model_id ||
      "本地模型"
    ).replace("Qwen/", "");
    $("voice-info").textContent =
      `识别：${health.asr_provider || "未配置"}；朗读：${health.speech_provider || "未配置"}。`;
    const select = $("model-mode"),
      modes = health.model_modes || [];
    const preferred = select.value || storage.get("modelMode", "base");
    select.replaceChildren();
    for (const mode of modes) {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = mode === "lora" ? "LoRA 实验版" : "基础模型";
      select.append(option);
    }
    select.value = modes.includes(preferred) ? preferred : "base";
    setBusy(state.busy);
    return health;
  } catch (error) {
    state.health = { ...state.health, status: "disconnected" };
    $("connection-label").textContent = "暂未连接本机服务";
    $("connection-dot").style.background = "#c3967e";
    throw error;
  }
}
function canNavigate() {
  if (state.busy || state.recording || state.recordPending) {
    toast("请先等待当前操作完成，录音时请先停止。");
    return false;
  }
  return true;
}
function setStatus(text) {
  $("chat-status").textContent = text;
}
function avatar(el, role) {
  el.className = "avatar " + (role?.color || "blue");
  el.replaceChildren();
  if (role && DESIGN[role.id]) {
    const img = document.createElement("img");
    img.src = "/ui/assets/portraits/" + DESIGN[role.id].image;
    img.alt = "";
    el.append(img);
  } else el.textContent = role?.name?.[0] || "问";
}
function formatTime(value) {
  const date = new Date(value);
  return (
    date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" }) +
    " " +
    date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
  );
}
const DESIGN = {
  companion: {
    image: "xiaoman.png",
  },
  study: {
    image: "zhixia.png",
  },
  interview: {
    image: "yanchuan.png",
  },
};
function roleDesign(role) {
  if (role?.profile && DESIGN[role.id]) return { ...role.profile, image: DESIGN[role.id].image };
  if (role?.custom) return { line: role.profile?.line || role.description, tags: [], scene: "原创角色", about: role.profile?.about || role.description, image: null };
  return null;
}
function identityFacts(identity) {
  if (!identity) return "";
  const facts = [
    ["性别年龄", `${identity.gender} · ${identity.age}`],
    ["所在地", identity.city],
    ["职业", identity.occupation],
    ["性格", identity.traits.join("、")],
    ["喜欢", identity.interests.join("、")],
    ["相处方式", identity.relationship],
    ["暂未设定", identity.unconfirmed.join("、")],
  ];
  return `<div class="character-facts">${facts
    .map(([label, value]) => `<div class="character-fact"><span>${label}</span><strong>${value}</strong></div>`)
    .join("")}</div>`;
}
let discoveryFilter = "all";
function renderCharacters() {
  const search = $("discover-search").value.trim().toLowerCase();
  const roles = state.characters.filter(
    (c) =>
      c.id !== "assistant" &&
      (discoveryFilter === "all" || c.id === discoveryFilter) &&
      [c.name, c.tag, c.description, ...(c.profile?.tags || [])]
        .join(" ")
        .toLowerCase()
        .includes(search),
  );
  $("character-grid").replaceChildren();
  roles.forEach((c) => {
    const d = roleDesign(c),
      card = document.createElement("button");
    card.className = "character-card";
    card.dataset.character = c.id;
    card.setAttribute("aria-label", `和${c.name}开始对话`);
    card.innerHTML = `<div class="portrait">${d.image ? `<img src="/ui/assets/portraits/${escapeHTML(d.image)}" alt="${escapeHTML(c.name)}的AI生成角色形象" loading="eager">` : '<div class="custom-role-art">✦</div>'}<span class="portrait-tag">${escapeHTML(c.tag)}</span><div class="portrait-shade"></div><div class="portrait-caption"><span class="portrait-scene">${escapeHTML(d.scene || "原创角色")}</span><h3>${escapeHTML(c.name)}<span class="card-enter">${icon("arrow")}</span></h3><p>${escapeHTML(d.line || c.description)}</p></div></div><div class="character-copy"><div class="character-tags">${d.tags.map((t) => `<span>${escapeHTML(t)}</span>`).join("")}</div><span class="card-ai">AI角色</span></div>`;
    $("character-grid").append(card);
  });
  $("character-count").textContent = `${roles.length}位原创角色`;
  $("no-characters").hidden = roles.length !== 0;
}
function renderProfile() {
  const c = character(state.current.character_id),
    d = roleDesign(c),
    panel = $("character-panel");
  panel.classList.remove("profile-open");
  panel.innerHTML = `<button class="icon-button close-profile" aria-label="关闭角色资料">${icon("close")}</button>${d?.image ? `<img class="profile-portrait" src="/ui/assets/portraits/${escapeHTML(d.image)}" alt="${escapeHTML(c.name)}的AI生成形象">` : '<div class="assistant-profile">✳</div>'}<div class="profile-copy"><span class="profile-label">角色资料 · AI生成形象</span><h2>${escapeHTML(c.name)}</h2><p class="profile-tagline">${escapeHTML(d?.line || c.description)}</p><div class="character-tags">${(d?.tags || ["原创", "本地角色"]).map((t) => `<span>${escapeHTML(t)}</span>`).join("")}</div>${c.identity ? `<div class="profile-divider"></div><h3>固定身份档案</h3>${identityFacts(c.identity)}` : ""}<div class="profile-divider"></div><h3>关于${escapeHTML(c.name)}</h3><p>${escapeHTML(d?.about || c.description || "一位由你创建的原创角色。")}</p><div class="profile-note">角色由AI扮演，可能产生不准确的回答。当前对话使用 ${escapeHTML(state.current.model_id?.replace("Qwen/", "") || "本地模型")}。</div></div>`;
  panel.querySelector(".close-profile").onclick = () =>
    panel.classList.remove("profile-open");
}
async function loadSessions() {
  state.sessions = (await request("/sessions?limit=50&nonempty=true")).sessions;
  renderSessions();
  await loadHistory();
}
async function loadHistory(append = false) {
  if (append && (state.historyLoading || !state.historyHasMore)) return;
  const version = ++state.historyVersion;
  const offset = append ? state.historyOffset : 0;
  const query = $("history-search").value.trim();
  state.historyLoading = true;
  $("history-more").disabled = true;
  $("history-status").textContent = "正在查找对话…";
  try {
    const result = await request(
      `/sessions?limit=20&offset=${offset}&nonempty=true&q=${encodeURIComponent(query)}`,
    );
    if (version !== state.historyVersion) return;
    const rows = result.sessions;
    state.historySessions = append
      ? [
          ...new Map(
            [...state.historySessions, ...rows].map((row) => [row.id, row]),
          ).values(),
        ]
      : rows;
    state.historyOffset = offset + rows.length;
    state.historyHasMore = result.has_more;
    renderHistory();
    $("history-status").textContent =
      `${query ? "搜索结果" : "已加载"} ${state.historySessions.length} 段对话${result.has_more ? "，还有更多" : "，已全部显示"}。`;
  } catch (error) {
    if (version === state.historyVersion)
      $("history-status").textContent =
        "历史未能更新，当前显示上次结果。" + error.message;
  } finally {
    if (version === state.historyVersion) {
      state.historyLoading = false;
      $("history-more").hidden = !state.historyHasMore;
      $("history-more").disabled = false;
    }
  }
}
function renderSessions() {
  const recent = $("recent-sessions");
  recent.replaceChildren();
  if (!state.sessions.length) {
    recent.innerHTML = '<p class="side-empty">你的对话会留在这里</p>';
  }
  state.sessions.slice(0, 8).forEach((s) => {
    const c = character(s.character_id),
      button = document.createElement("button");
    button.className =
      "session-link" + (state.current?.id === s.id ? " selected" : "");
    const av = document.createElement("span");
    avatar(av, c);
    const text = document.createElement("div"),
      title = document.createElement("p"),
      time = document.createElement("span");
    title.className = "session-title";
    title.textContent = s.title;
    time.className = "session-time";
    time.textContent = c.name + " · " + formatTime(s.updated_at);
    text.append(title, time);
    button.append(av, text);
    button.addEventListener("click", () => openSession(s.id));
    recent.append(button);
  });
  renderHistory();
}
function renderHistory() {
  const filter = $("history-search").value.trim().toLowerCase(),
    list = $("history-list");
  list.replaceChildren();
  const rows = state.historySessions;
  if (!rows.length) {
    const el = document.createElement("div");
    el.className = "empty-state";
    el.textContent = filter
      ? "没有找到这段对话，换个关键词试试。"
      : "还没有对话。去发现页选一个角色吧。";
    list.append(el);
  }
  rows.forEach((s) => {
    const c = character(s.character_id),
      row = document.createElement("div"),
      b = document.createElement("button");
    row.className = "history-row";
    b.className = "history-card";
    const av = document.createElement("span");
    avatar(av, c);
    const content = document.createElement("div"),
      h = document.createElement("h3"),
      p = document.createElement("p");
    h.textContent = s.title;
    p.textContent =
      c.name +
      " · " +
      (s.model_mode === "lora" ? "LoRA 实验版" : "原始模型") +
      " · " +
      formatTime(s.updated_at);
    content.append(h, p);
    b.append(av, content);
    b.insertAdjacentHTML("beforeend", icon("arrow"));
    b.addEventListener("click", () => openSession(s.id));
    const remove = document.createElement("button");
    remove.className = "history-delete";
    remove.type = "button";
    remove.textContent = "删除";
    remove.setAttribute("aria-label", `删除与${c.name}的对话：${s.title}`);
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      askDeleteSession(s.id);
    });
    row.append(b, remove);
    list.append(row);
  });
}
function askDeleteSession(id) {
  const row = state.historySessions.find((item) => item.id === id) ||
    state.sessions.find((item) => item.id === id);
  if (!row || !canNavigate()) return;
  state.pendingDeleteSessionId = id;
  $("delete-session-title").textContent = row.title || "新对话";
  $("delete-session-dialog").showModal();
}
function showView(view) {
  $("discover-view").hidden = view !== "discover";
  $("history-view").hidden = view !== "history";
  $("chat-view").hidden = view !== "chat";
  $("nav-discover").classList.toggle("active", view === "discover");
  $("nav-history").classList.toggle("active", view !== "discover");
  $("breadcrumb").textContent =
    view === "discover"
      ? "发现 / 适合此刻的对话"
      : view === "history"
        ? "对话 / 留住每一次交流"
        : "对话 / " + character(state.current?.character_id).name;
}
function stopAudio() {
  const player = $("speech-player");
  player.pause();
  player.hidden = true;
  player.removeAttribute("src");
  if (state.audioURL) {
    URL.revokeObjectURL(state.audioURL);
    state.audioURL = null;
  }
}
function activateSession(current, clearDraft = false) {
  if (!clearDraft) saveDraft();
  state.current = current;
  $("model-caption").textContent = (
    current.model_id ||
    state.health.model_id ||
    "本地模型"
  ).replace("Qwen/", "");
  state.voiceEpoch++;
  state.voiceFile = null;
  if (clearDraft) saveDraft(current.id, "");
  const pending = storage.get("pendingDraft." + current.id, "");
  const last = current.messages.slice(-2);
  const pendingSaved =
    pending &&
    last.length === 2 &&
    last[0].role === "user" &&
    last[0].content === pending.trim() &&
    last[1].role === "assistant";
  if (clearDraft || pendingSaved) storage.set("pendingDraft." + current.id, "");
  $("draft").value = clearDraft
    ? ""
    : storage.get("draft." + current.id, "") || (pendingSaved ? "" : pending);
  state.draftVersion++;
  $("retry-message").hidden = !pending || !!pendingSaved || clearDraft;
  stopAudio();
  setStatus(
    current.model_id && current.model_id !== state.health.model_id
      ? "这段历史使用旧模型，记录已保留。点击“新对话”可使用当前模型。"
      : "",
  );
  const c = character(current.character_id);
  $("chat-name").textContent = c.name;
  $("chat-tag").textContent = c.tag + " · AI角色";
  $("edit-character-card").hidden = !c.custom;
  avatar($("chat-avatar"), c);
  renderProfile();
  showView("chat");
  renderMessages();
  renderSessions();
  storage.set("lastSession", current.id);
}
async function openSession(id) {
  if (!canNavigate()) return;
  setBusy("navigation");
  try {
    activateSession(await request("/sessions/" + id));
  } catch (e) {
    toast(e.message);
  } finally {
    setBusy(null);
  }
}
async function newSession(id) {
  if (!canNavigate()) return;
  setBusy("navigation");
  try {
    if ((await refreshHealth()).status !== "ready")
      throw new Error("模型尚未就绪，请稍后再试；历史仍可查看。");
    activateSession(
      await jsonPost("/sessions", {
        character_id: id,
        model_mode: $("model-mode").value,
      }),
    );
    await loadSessions();
    $("draft").focus();
  } catch (e) {
    toast(e.message);
  } finally {
    setBusy(null);
  }
}
function renderText(el, text) {
  el.replaceChildren();
  const chunks = text.split(/(```[\s\S]*?```)/g);
  chunks.forEach((chunk) => {
    if (chunk.startsWith("```") && chunk.endsWith("```")) {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = chunk.slice(3, -3).replace(/^\w*\n/, "");
      pre.append(code);
      el.append(pre);
    } else {
      el.append(document.createTextNode(chunk));
    }
  });
}
function messageNode(m, temporary = false) {
  const c = character(state.current.character_id),
    item = document.createElement("article");
  item.className = "message " + (m.role === "user" ? "user" : "assistant");
  const av = document.createElement("span");
  avatar(av, m.role === "user" ? { name: "我", color: "user" } : c);
  const body = document.createElement("div");
  body.className = "message-body";
  const name = document.createElement("p");
  name.className = "message-name";
  name.textContent = m.role === "user" ? "我" : c.name;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  renderText(bubble, m.content);
  body.append(name, bubble);
  if (m.role === "assistant" && !temporary && m.id) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const copy = document.createElement("button");
    copy.className = "icon-button";
    copy.innerHTML = icon("copy");
    copy.title = "复制回答";
    copy.setAttribute("aria-label", "复制这条回答");
    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(m.content);
        toast("已复制回答");
      } catch {
        toast("无法访问剪贴板，请手动选择文字复制。");
      }
    };
    const read = document.createElement("button");
    read.className = "icon-button";
    read.innerHTML = icon("volume");
    read.title = "朗读回答";
    read.setAttribute("aria-label", "朗读这条回答");
    read.disabled = !state.health.speech_provider;
    read.onclick = () => speak(m.id);
    actions.append(copy, read);
    body.append(actions);
  }
  item.append(av, body);
  return { item, bubble };
}
function renderMessages() {
  const box = $("messages");
  box.replaceChildren();
  const c = character(state.current.character_id);
  if (!state.current.messages.length) {
    const intro = document.createElement("div");
    intro.className = "greeting";
    const av = document.createElement("div");
    avatar(av, c);
    const h = document.createElement("h2");
    h.textContent = "和" + c.name + "聊聊";
    const p = document.createElement("p");
    p.textContent = c.greeting;
    const label = document.createElement("div");
    label.className = "greeting-label";
    label.textContent = "预设开场白 · AI角色";
    intro.append(av, h, p, label);
    const starters = document.createElement("div");
    starters.className = "starters";
    c.starters.forEach((text) => {
      const b = document.createElement("button");
      b.className = "starter";
      b.textContent = text;
      b.onclick = () => {
        $("draft").value = text;
        state.draftVersion++;
        saveDraft();
        $("draft").focus();
      };
      starters.append(b);
    });
    intro.append(starters);
    box.append(intro);
  } else {
    state.current.messages.forEach((m) => box.append(messageNode(m).item));
  }
  box.scrollTop = box.scrollHeight;
  state.autoScroll = true;
}
function humanMetrics(m) {
  let text = `已保存 · ${Number(m.seconds || 0).toFixed(1)}秒`;
  if (m.context_omitted_turns)
    text += `。本次省略最早${m.context_omitted_turns}轮，完整记录仍在；引用旧内容请重新说明。`;
  if (m.ended_with_eos === false)
    text += " 回答可能未完，可发送“请继续刚才的回答”，或在设置中增加长度。";
  return text;
}
async function* sse(response) {
  // 网络块不等于SSE事件；保留UTF-8解码状态和未结束的事件片段。
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done
        ? decoder.decode()
        : decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        let type = "message";
        const data = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) type = line.slice(6).trim();
          if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        }
        if (data.length) yield { type, data: JSON.parse(data.join("\n")) };
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}
async function send(retrying = false) {
  if (state.busy) return;
  const text = $("draft").value.trim();
  retrying =
    retrying ||
    (!!text &&
      storage.get("pendingDraft." + state.current?.id, "").trim() === text);
  if (!text) {
    toast("先输入想聊的内容吧。");
    return;
  }
  if (text.length > 4096) {
    toast("这次输入太长，请缩短到4096个字符以内。");
    return;
  }
  if (!state.current) {
    toast("请先选择一个角色。");
    return;
  }
  stopAudio();
  state.stopRequested = false;
  const id = state.current.id,
    before = state.current.messages.slice(),
    submittedDraft = $("draft").value,
    submittedVersion = state.draftVersion;
  // 等待响应期间仍允许准备下一条消息，只清除本次真正提交的草稿。
  const clearSubmittedDraft = () => {
    if (
      state.draftVersion === submittedVersion &&
      $("draft").value === submittedDraft
    ) {
      $("draft").value = "";
      state.draftVersion++;
      saveDraft();
    }
  };
  let completed = false,
    reply = "",
    hadError = false;
  setBusy("chat");
  setStatus(retrying ? "先检查上一轮是否已保存…" : "正在想一想…");
  $("retry-message").hidden = true;
  try {
    const health = await refreshHealth();
    if (health.status !== "ready")
      throw new Error("模型尚未就绪，请稍后重试。");
    if (state.current.model_id && state.current.model_id !== health.model_id)
      throw new Error("服务已换用其他模型，请新建对话；原草稿已保留。");
    if (retrying) {
      const saved = await request("/sessions/" + id);
      const last = saved.messages.slice(-2);
      if (
        last.length === 2 &&
        last[0].role === "user" &&
        last[0].content === text &&
        last[1].role === "assistant"
      ) {
        state.current = saved;
        renderMessages();
        clearSubmittedDraft();
        storage.set("pendingDraft." + id, "");
        setStatus("已恢复这个问题的保存回答，没有重复发送。");
        return;
      }
    }
    const limit = Number($("answer-length").value);
    storage.set("pendingDraft." + id, submittedDraft);
    await withResponse(
      "/chat/stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: id,
          message: text,
          max_new_tokens: limit,
          timeout_seconds: limit > 512 ? 240 : 120,
        }),
      },
      async (response) => {
        if (!response.ok) {
          let error;
          try {
            error = (await response.json()).detail;
          } catch {}
          throw new Error(error || "暂时无法生成，请稍后重试。");
        }
        clearSubmittedDraft();
        const box = $("messages");
        if (!before.length) box.replaceChildren();
        box.append(messageNode({ role: "user", content: text }, true).item);
        const live = messageNode({ role: "assistant", content: "…" }, true);
        box.append(live.item);
        box.scrollTop = box.scrollHeight;
        for await (const event of sse(response)) {
          if (event.type === "start" && state.stopRequested) {
            await jsonPost("/sessions/" + id + "/stop", {});
          }
          if (event.type === "delta") {
            reply += event.data.text;
            renderText(live.bubble, reply);
            if (state.autoScroll) box.scrollTop = box.scrollHeight;
          } else if (event.type === "done") {
            completed = true;
            storage.set("pendingDraft." + id, "");
            state.current = await request("/sessions/" + id);
            renderMessages();
            setStatus(humanMetrics(event.data.metrics));
          } else if (event.type === "error") {
            throw new Error(event.data.detail || "本轮没有完成。");
          }
        }
        if (!completed) throw new Error("连接中断，正在核对保存记录。");
      },
      (limit > 512 ? 240 : 120) * 1000 + 30000,
    );
  } catch (error) {
    hadError = true;
    let recovered = false;
    try {
      const saved = await request("/sessions/" + id),
        rows = saved.messages;
      recovered =
        rows.length === before.length + 2 &&
        rows.slice(0, before.length).every((m, i) => m.id === before[i].id) &&
        rows.at(-2).role === "user" &&
        rows.at(-2).content === text &&
        rows.at(-1).role === "assistant";
      state.current = saved;
      renderMessages();
    } catch {}
    if (recovered) {
      storage.set("pendingDraft." + id, "");
      setStatus("连接有波动，但回答已保存，已恢复显示。");
    } else {
      if (!$("draft").value) {
        $("draft").value = text;
        state.draftVersion++;
        saveDraft();
      }
      $("retry-message").hidden = false;
      setStatus(error.message + " 问题已保留，可修改后重试。");
    }
  } finally {
    setBusy(null);
    try {
      await loadSessions();
    } catch {
      if (!hadError) toast("回答已完成，但历史列表暂时未刷新。");
    }
  }
}
async function stopGeneration() {
  if (state.busy !== "chat" || !state.current) return;
  state.stopRequested = true;
  try {
    await jsonPost("/sessions/" + state.current.id + "/stop", {});
    setStatus("正在停止，等待当前步骤退出；未完成问答不会保存。");
  } catch (e) {
    toast(e.message);
  }
}
async function speak(messageId) {
  if (!state.current || !canNavigate()) return;
  const id = state.current.id;
  setBusy("speech");
  setStatus("正在生成语音，文字回答已保存…");
  try {
    await withResponse(
      `/sessions/${id}/speech`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: messageId }),
      },
      async (response) => {
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "朗读失败");
        }
        if (response.headers.get("X-Message-ID") !== String(messageId))
          throw new Error("音频与回答不匹配");
        stopAudio();
        state.audioURL = URL.createObjectURL(await response.blob());
        $("speech-player").src = state.audioURL;
        $("speech-player").hidden = false;
        setStatus("语音已生成，点击播放器收听或暂停。");
      },
      180000,
    );
  } catch (e) {
    setStatus("语音未完成，文字仍保留。" + e.message);
  } finally {
    setBusy(null);
  }
}
function resetVoice() {
  state.voiceEpoch++;
  state.voiceFile = null;
  $("audio-file").value = "";
  $("audio-file-label").textContent = "或选择本机音频，推荐 PCM WAV";
  $("voice-status").textContent = "识别可能漏字，请检查；麦克风权限由你决定。";
  $("transcribe-audio").disabled = true;
}
function releaseRecorder(discard = false) {
  if (state.recordTimer) {
    clearInterval(state.recordTimer);
    state.recordTimer = null;
  }
  if (discard) state.voiceEpoch++;
  state.recordPending = false;
  $("record-audio").disabled = !!state.busy;
  if (state.recording && state.recording.state !== "inactive")
    state.recording.stop();
  state.recordStream?.getTracks().forEach((t) => t.stop());
  state.recordStream = null;
  state.recording = null;
  $("record-audio").classList.remove("recording");
  $("record-audio").setAttribute("aria-label", "开始录音");
  $("record-label").textContent = "点击开始录音";
}
async function record() {
  if (state.recording) {
    releaseRecorder();
    return;
  }
  if (state.busy || state.recordPending) return;
  const epoch = ++state.voiceEpoch;
  state.recordPending = true;
  state.voiceFile = null;
  $("transcribe-audio").disabled = true;
  $("audio-file-label").textContent = "正在准备新录音…";
  $("record-audio").disabled = true;
  $("record-label").textContent = "等待麦克风权限…";
  let stream;
  try {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error("当前浏览器不支持录音，请使用Chrome/Edge或上传音频。");
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (epoch !== state.voiceEpoch || !$("voice-dialog").open || state.busy) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    const mime = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
    ].find((t) => MediaRecorder.isTypeSupported(t));
    const recorder = mime
      ? new MediaRecorder(stream, { mimeType: mime })
      : new MediaRecorder(stream);
    const chunks = [];
    state.recordStream = stream;
    state.recording = recorder;
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = () => {
      if (epoch !== state.voiceEpoch || !$("voice-dialog").open) return;
      // 设备自行结束录音时也释放计时器和麦克风。
      if (state.recording === recorder) releaseRecorder();
      state.voiceFile = new Blob(chunks, { type: recorder.mimeType });
      $("audio-file-label").textContent = "录音已准备好";
      $("voice-status").textContent = "点击识别，结果会先放入输入框。";
      $("transcribe-audio").disabled = false;
    };
    recorder.onerror = () => {
      if (epoch !== state.voiceEpoch) return;
      releaseRecorder(true);
      $("voice-status").textContent =
        "录音设备发生错误，请重新录音或上传音频。";
    };
    recorder.start();
    const start = Date.now();
    $("record-audio").classList.add("recording");
    $("record-audio").setAttribute("aria-label", "停止录音");
    $("record-label").textContent = "正在录音，点击结束";
    state.recordTimer = setInterval(() => {
      const seconds = Math.floor((Date.now() - start) / 1000);
      $("record-time").textContent =
        `00:${String(Math.min(seconds, 30)).padStart(2, "0")} / 00:30`;
      if (seconds >= 30) releaseRecorder();
    }, 200);
  } catch (e) {
    stream?.getTracks().forEach((t) => t.stop());
    // 上次权限请求晚到的失败，不得停止后来开始的新录音。
    if (epoch !== state.voiceEpoch) return;
    releaseRecorder(true);
    $("voice-status").textContent =
      e.name === "NotAllowedError"
        ? "未获得麦克风权限，可以上传音频。"
        : e.name === "NotFoundError"
          ? "没有找到麦克风，可以上传PCM WAV音频。"
          : e.message;
  } finally {
    if (epoch === state.voiceEpoch) {
      state.recordPending = false;
      $("record-audio").disabled = !!state.busy;
    }
  }
}
async function transcribe() {
  if (!state.voiceFile || state.busy || state.recording || state.recordPending)
    return;
  const epoch = state.voiceEpoch,
    sid = state.current?.id,
    version = state.draftVersion;
  setBusy("asr");
  $("voice-status").textContent = "正在本机识别，请稍等…";
  try {
    if (state.voiceFile.size > 8 * 1024 * 1024)
      throw new Error("音频超过8MiB，请选一段更短的录音。");
    const result = await request(
      "/transcribe",
      {
        method: "POST",
        headers: { "Content-Type": state.voiceFile.type || "application/octet-stream" },
        body: state.voiceFile,
      },
      180000,
    );
    if (epoch !== state.voiceEpoch || sid !== state.current?.id) return;
    if (version !== state.draftVersion) {
      $("voice-status").textContent =
        "你已修改草稿，识别结果未覆盖。结果：" + result.text;
      return;
    }
    $("draft").value = result.text;
    state.draftVersion++;
    saveDraft();
    $("voice-dialog").close();
    $("draft").focus();
    setStatus("已转为文字，请检查、修改后发送。");
  } catch (e) {
    $("voice-status").textContent = "识别未完成，原草稿保留。" + e.message;
  } finally {
    setBusy(null);
  }
}
function exportChat() {
  if (!state.current || !canNavigate()) return;
  const blob = new Blob(
    [
      JSON.stringify(
        {
          app: "拾语",
          exported_at: new Date().toISOString(),
          session: state.current,
        },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = `拾语-${state.current.id.slice(0, 8)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("已导出当前对话。");
}
async function initialize() {
  icons();
  $("answer-length").value = storage.get("answerLength", "512");
  try {
    const [, roles] = await Promise.all([
      refreshHealth(),
      request("/characters"),
    ]);
    state.characters = roles.characters;
    renderCharacters();
    setBusy(null);
    await loadSessions();
    const last = storage.get("lastSession", "");
    if (last && !state.current) await openSession(last);
  } catch (e) {
    $("connection-label").textContent = "暂未连接本机服务";
    $("connection-dot").style.background = "#c3967e";
    $("model-info").textContent = e.message;
    toast("本机服务暂不可用，请确认后端已启动，然后刷新页面。");
  }
}
function showCharacterDraft(card) {
  $("card-name").value = card.name || "";
  $("card-tagline").value = card.tagline || "";
  $("card-description").value = card.description || "";
  $("card-background").value = card.background || "";
  $("card-personality").value = card.personality || "";
  $("card-speech-style").value = card.speech_style || "";
  $("card-relationship").value = card.relationship || "";
  $("card-greeting").value = card.greeting || "";
  $("card-examples").value = (card.example_dialogue || []).map((row) => `${row.user}｜${row.character}`).join("\n");
  $("card-boundaries").value = card.boundaries || "";
  $("character-draft-notice").textContent = card.draft_notice || "生成内容为草稿，请检查后保存。";
  $("card-facts").value = (card.fixed_facts || []).map((fact) =>
    `${fact.status === "confirmed" ? "确认" : "待确认"}｜${fact.key}｜${fact.value}`,
  ).join("\n");
  $("card-worldbook").value = (card.worldbook || []).map((entry) =>
    `${(entry.keywords || []).join(",")}｜${entry.title}｜${entry.content}`,
  ).join("\n");
  state.characterDraft = card;
  $("character-draft-fields").hidden = false;
}
function readCharacterDraft() {
  const facts = $("card-facts").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line) => {
    const parts = line.split("｜");
    if (parts.length < 3) throw new Error("固定设定请按“确认/待确认｜名称｜内容”填写。" );
    const status = parts.shift().trim() === "确认" ? "confirmed" : "proposal";
    return { status, key: parts.shift().trim(), value: parts.join("｜").trim() };
  });
  const worldbook = $("card-worldbook").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line, index) => {
    const parts = line.split("｜");
    if (parts.length < 3) throw new Error("世界书请按“关键词1,关键词2｜标题｜内容”填写。" );
    return { id: `entry-${index + 1}`, keywords: parts.shift().split(/[，,]/).map((word) => word.trim()).filter(Boolean), title: parts.shift().trim(), content: parts.join("｜").trim(), enabled: true };
  });
  const example_dialogue = $("card-examples").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line) => {
    const parts = line.split("｜");
    if (parts.length < 2) throw new Error("示例对话请按“用户输入｜角色完整回复”填写。" );
    return { user: parts.shift().trim(), character: parts.join("｜").trim() };
  });
  return {
    ...state.characterDraft,
    name: $("card-name").value.trim(), tagline: $("card-tagline").value.trim(),
    description: $("card-description").value.trim(), background: $("card-background").value.trim(),
    personality: $("card-personality").value.trim(), speech_style: $("card-speech-style").value.trim(),
    relationship: $("card-relationship").value.trim(), greeting: $("card-greeting").value.trim(), example_dialogue,
    boundaries: $("card-boundaries").value.trim(), fixed_facts: facts, worldbook,
  };
}
$("create-character").onclick = () => {
  state.characterDraft = null;
  state.editingCharacterId = null;
  $("character-inspiration").value = "";
  $("character-generation-status").textContent = "";
  $("character-draft-fields").hidden = true;
  $("character-editor-dialog").showModal();
};
$("edit-character-card").onclick = async () => {
  const id = state.current?.character_id;
  if (!id) return;
  try {
    const result = await request(`/characters/${encodeURIComponent(id)}`);
    state.editingCharacterId = id;
    $("character-generation-status").textContent = "正在编辑本机保存的角色卡。";
    $("character-inspiration").value = result.card.source_notes || "";
    showCharacterDraft(result.card);
    $("character-editor-dialog").showModal();
  } catch (error) { toast(error.message); }
};
$("generate-character").onclick = async () => {
  const inspiration = $("character-inspiration").value.trim();
  if (!inspiration) { $("character-generation-status").textContent = "先写几条关键词或背景。"; return; }
  const button = $("generate-character");
  button.disabled = true;
  $("character-generation-status").textContent = "正在让本机 Qwen 整理角色卡，可能需要几十秒……";
  try {
    const result = await jsonPost("/characters/generate", { inspiration }, 120000);
    if (state.editingCharacterId) result.card.id = state.editingCharacterId;
    showCharacterDraft(result.card);
    $("character-generation-status").textContent = "草稿已生成。检查固定事实和世界书后再保存。";
  } catch (error) {
    $("character-generation-status").textContent = error.message;
  } finally { button.disabled = false; }
};
$("save-character").onclick = async () => {
  const button = $("save-character");
  button.disabled = true;
  try {
    const result = await jsonPost("/characters", { card: readCharacterDraft() });
    const roles = await request("/characters");
    state.characters = roles.characters;
    renderCharacters();
    $("character-editor-dialog").close();
    state.editingCharacterId = null;
    toast(`${result.card.name} 已保存为本地角色卡。`);
    await newSession(result.card.id);
  } catch (error) {
    toast(error.message);
  } finally { button.disabled = false; }
};
document.addEventListener("click", (event) => {
  const roleButton = event.target.closest("[data-character]");
  if (roleButton) {
    newSession(roleButton.dataset.character);
    return;
  }
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "settings") {
    $("settings-dialog").showModal();
  } else if (action && canNavigate()) {
    showView(action);
  }
  const close = event.target.closest("[data-close]");
  if (close) $(close.dataset.close).close();
});
$("send-message").onclick = () => send();
$("retry-message").onclick = () => send(true);
$("stop-message").onclick = stopGeneration;
$("draft").oninput = () => {
  state.draftVersion++;
  saveDraft();
};
$("draft").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    send();
  }
};
$("messages").onscroll = () => {
  const el = $("messages");
  state.autoScroll = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
};
$("new-role-chat").onclick = () =>
  newSession(
    state.characters.some((c) => c.id === state.current?.character_id)
      ? state.current.character_id
      : "interview",
  );
$("export-chat").onclick = exportChat;
let historySearchTimer;
$("history-search").oninput = () => {
  clearTimeout(historySearchTimer);
  // 立即使旧请求失效，不能等到防抖结束后才丢弃旧搜索结果。
  state.historyVersion++;
  state.historyLoading = false;
  state.historyHasMore = false;
  state.historySessions = [];
  $("history-more").hidden = true;
  $("history-list").replaceChildren();
  $("history-status").textContent = "正在查找对话…";
  historySearchTimer = setTimeout(() => loadHistory(), 250);
};
$("history-more").onclick = () => loadHistory(true);
$("refresh-history").onclick = async () => {
  if (!canNavigate()) return;
  setBusy("navigation");
  try {
    await refreshHealth();
    if (!state.characters.length) {
      state.characters = (await request("/characters")).characters;
      renderCharacters();
    }
    await loadSessions();
    toast("历史已刷新。");
  } catch (e) {
    toast(e.message);
  } finally {
    setBusy(null);
  }
};
$("answer-length").onchange = () =>
  storage.set("answerLength", $("answer-length").value);
$("model-mode").onchange = () =>
  storage.set("modelMode", $("model-mode").value);
$("open-voice").onclick = () => {
  if (!state.current || !canNavigate()) return;
  resetVoice();
  $("voice-dialog").showModal();
};
$("voice-dialog").addEventListener("close", () => releaseRecorder(true));
$("record-audio").onclick = record;
$("audio-file").onchange = () => {
  if (state.recording || state.recordPending) releaseRecorder(true);
  state.voiceEpoch++;
  state.voiceFile = $("audio-file").files[0] || null;
  $("audio-file-label").textContent = state.voiceFile?.name || "请选择音频";
  $("transcribe-audio").disabled = !state.voiceFile || !!state.busy;
};
$("transcribe-audio").onclick = transcribe;
window.addEventListener("beforeunload", () => {
  saveDraft();
  releaseRecorder(true);
  stopAudio();
});
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  state.installPrompt = e;
});
$("install-app").onclick = async () => {
  if (state.installPrompt) {
    await state.installPrompt.prompt();
    state.installPrompt = null;
  } else {
    toast(
      "可在Chrome/Edge菜单中选择“安装此页面为应用”或“创建快捷方式”。服务需在本机运行。",
    );
  }
};
$("confirm-clear").onclick = async () => {
  if (!state.current || !canNavigate()) return;
  const id = state.current.id;
  setBusy("clearing");
  try {
    const cleared = await request("/sessions/" + id + "/messages", {
      method: "DELETE",
    });
    $("clear-dialog").close();
    activateSession(cleared, true);
    setStatus("本会话消息已清空。");
    await loadSessions();
  } catch (e) {
    toast(e.message);
  } finally {
    setBusy(null);
  }
};
$("confirm-delete-session").onclick = async () => {
  const id = state.pendingDeleteSessionId;
  if (!id || !canNavigate()) return;
  const deletingCurrent = state.current?.id === id;
  const previous = state.current;
  setBusy("deleting");
  try {
    await request("/sessions/" + id, { method: "DELETE" });
    $("delete-session-dialog").close();
    state.pendingDeleteSessionId = null;
    state.sessions = state.sessions.filter((item) => item.id !== id);
    state.historySessions = state.historySessions.filter((item) => item.id !== id);
    storage.remove("draft." + id);
    storage.remove("pendingDraft." + id);
    let restartError = "";
    if (deletingCurrent) {
      stopAudio();
      state.current = null;
      showView("discover");
      const characterId = state.characters.some((c) => c.id === previous.character_id)
        ? previous.character_id : "interview";
      const modes = state.health.model_modes || [];
      const modelMode = previous.model_id === state.health.model_id && modes.includes(previous.model_mode)
        ? previous.model_mode : "base";
      try {
        const fresh = await jsonPost("/sessions", { character_id: characterId, model_mode: modelMode });
        activateSession(fresh, true);
      } catch (error) {
        restartError = " 新对话未能自动开启：" + error.message;
      }
    }
    await loadSessions();
    toast("这段对话及其消息已删除。" + restartError);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(null);
  }
};
const clearButton = document.createElement("button");
clearButton.className = "text-button";
clearButton.textContent = "清空当前会话";
clearButton.onclick = () => {
  if (!state.current) {
    toast("请先进入一段对话。");
    return;
  }
  if (!canNavigate()) return;
  $("settings-dialog").close();
  $("clear-dialog").showModal();
};
document.querySelector(".settings-links").append(clearButton);
async function editVoiceProfile() {
  const characterId = state.current?.character_id;
  if (!characterId || !canNavigate()) return;
  try {
    const result = await request(`/characters/${encodeURIComponent(characterId)}/voice-profile`);
    state.voiceProfileCharacterId = characterId;
    $("voice-profile-heading").textContent = `${character(state.current.character_id).name} · 音色档案`;
    $("voice-profile-name").value = result.display_name || `${character(state.current.character_id).name} · 日常音色`;
    $("voice-profile-style").value = result.voice_style || "";
    $("voice-profile-text").value = result.reference_text || "";
    $("voice-profile-audio").value = "";
    $("voice-profile-status").textContent = result.has_reference_audio
      ? "已保存参考音频。更换音频时请同步更新逐字稿。"
      : "尚未上传参考音频；可先保存档案，上传音频后才能由 GPT-SoVITS 使用。";
    $("voice-profile-dialog").showModal();
  } catch (error) { toast(error.message); }
}
$("configure-current-voice").onclick = editVoiceProfile;
$("save-voice-profile").onclick = async () => {
  const characterId = state.voiceProfileCharacterId;
  if (!characterId) return;
  const button = $("save-voice-profile"), file = $("voice-profile-audio").files?.[0];
  button.disabled = true;
  $("voice-profile-status").textContent = "正在保存音色档案…";
  try {
    await request(`/characters/${encodeURIComponent(characterId)}/voice-profile`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        display_name: $("voice-profile-name").value.trim(),
        voice_style: $("voice-profile-style").value.trim(),
        reference_text: $("voice-profile-text").value.trim(),
        language: "zh",
      }),
    });
    if (file) {
      if (!file.type.includes("wav") && !file.name.toLowerCase().endsWith(".wav"))
        throw new Error("请选择 WAV 格式的参考音频。");
      await request(`/characters/${encodeURIComponent(characterId)}/voice-profile/reference-audio`, {
        method: "PUT", headers: { "Content-Type": file.type || "audio/wav" }, body: file,
      }, 30000);
    }
    const updated = await request(`/characters/${encodeURIComponent(characterId)}/voice-profile`);
    $("voice-profile-status").textContent = updated.gpt_sovits_ready
      ? "音色档案已完整保存；启动 GPT-SoVITS 后，播放此角色的回答会使用这份参考音频。"
      : "档案已保存。还需要填写逐字稿并上传 PCM WAV，GPT-SoVITS 才能使用它。";
    if (updated.gpt_sovits_ready) {
      $("voice-profile-dialog").close();
      toast("角色音色档案已保存。当前语音引擎若仍是 MeloTTS，不会使用该音色。 ");
    }
  } catch (error) {
    $("voice-profile-status").textContent = error.message;
  } finally { button.disabled = false; }
};
$("discover-search").oninput = renderCharacters;
document.querySelectorAll("[data-filter]").forEach(
  (b) =>
    (b.onclick = () => {
      discoveryFilter = b.dataset.filter;
      document
        .querySelectorAll("[data-filter]")
        .forEach((t) => t.classList.toggle("active", t === b));
      renderCharacters();
    }),
);
$("toggle-profile").onclick = () =>
  $("character-panel").classList.toggle("profile-open");
initialize();
