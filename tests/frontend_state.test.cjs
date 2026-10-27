// Execute the shipped frontend with a minimal DOM, delaying real handler boundaries.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function harness(savedStorage = new Map()) {
  const nodes = new Map();
  function element() {
    return { value: '', hidden: false, dataset: {}, style: {}, children: [], listeners: {},
      classList: { add() {}, remove() {}, toggle() {} },
      append(...items) { this.children.push(...items); }, replaceChildren(...items) { this.children = items; },
      insertAdjacentHTML() {}, setAttribute() {},
      addEventListener(name, fn) { this.listeners[name] = fn; }, close() { this.open = false; },
      showModal() { this.open = true; }, focus() {} };
  }
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: node, createElement: element,
      querySelector: node, querySelectorAll: () => [], addEventListener() {} },
    window: { addEventListener() {} }, localStorage: {
      getItem: key => savedStorage.get(key) ?? null,
      setItem: (key, value) => savedStorage.set(key, String(value)),
      removeItem: key => savedStorage.delete(key),
    },
    setTimeout, clearTimeout, setInterval, clearInterval, TextDecoder, URL, console, AbortController, Blob,
  });
  const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
  vm.runInContext(source.replace(/initialize\(\);\s*$/, ''), context);
  vm.runInContext(`
    const originalRequest = request;
    const originalHealth = refreshHealth;
    const originalWithResponse = withResponse;
    const originalSse = sse;
    const originalActivate = activateSession;
    state.current = { id: 'one', model_id: 'model', messages: [] };
    state.health = { model_id: 'model', status: 'ready' };
    refreshHealth = async () => state.health;
    stopAudio = () => {};
    renderMessages = () => {};
    loadSessions = async () => {};
    messageNode = () => ({ item: {}, bubble: {} });
    activateSession = session => { state.current = session; };
    sse = async function* () { yield { type: 'done', data: { metrics: {} } }; };
    request = async () => ({ id: 'one', model_id: 'model', messages: [] });
  `, context);
  return { node, context, savedStorage, run: code => vm.runInContext(code, context) };
}
function deferred() {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
}

test('new chat keeps a visible role and falls back to Yanchuan for legacy hidden roles', () => {
  const h = harness();
  h.run(`state.characters = [{id:'interview'}, {id:'companion'}];
    state.current = {id:'old', character_id:'study'};
    newSession = async id => { state.testNewRole = id; };
    $('new-role-chat').onclick()`);
  assert.equal(h.run('state.testNewRole'), 'interview');
  h.run(`state.current = {id:'visible', character_id:'companion'}; $('new-role-chat').onclick()`);
  assert.equal(h.run('state.testNewRole'), 'companion');
});

test('editing draft while stream headers are pending keeps the new draft', async () => {
  const h = harness(), pending = deferred();
  h.context.fetch = () => pending.promise;
  h.node('draft').value = '第一条问题';
  const sending = h.run('send()');
  h.node('draft').value = '正在准备的下一条';
  h.node('draft').oninput();
  pending.resolve({ ok: true });
  await sending;
  assert.equal(h.node('draft').value, '正在准备的下一条');
});

test('unchanged submitted draft is cleared', async () => {
  const h = harness();
  h.context.fetch = async () => ({ ok: true });
  h.node('draft').value = '第一条问题';
  await h.run('send()');
  assert.equal(h.node('draft').value, '');
});

test('audio transcription sends the selected recording MIME type', async () => {
  const h = harness();
  h.run(`state.voiceFile = new Blob(['webm'], { type: 'audio/webm;codecs=opus' });
    state.draftVersion = 0;`);
  h.context.calls = [];
  h.run(`request = async (url, options) => { calls.push([url, options]); return { text: '识别结果' }; }`);
  await h.run('transcribe()');
  assert.equal(h.context.calls[0][0], '/transcribe');
  assert.equal(h.context.calls[0][1].headers['Content-Type'], 'audio/webm;codecs=opus');
  assert.equal(h.node('draft').value, '识别结果');
});

test('history exposes an explicit delete action and deletes the selected session', async () => {
  const h = harness();
  h.run(`state.characters = [{ id: 'companion', name: '小满', color: 'peach' }];
    state.historySessions = [{ id: 'old-session', character_id: 'companion', title: '旧聊天',
      model_mode: 'base', updated_at: '2026-10-25T00:00:00Z' }];
    renderHistory();`);
  const row = h.node('history-list').children[0];
  const remove = row.children[1];
  await remove.listeners.click({ stopPropagation() {} });
  assert.equal(h.node('delete-session-title').textContent, '旧聊天');
  assert.equal(h.node('delete-session-dialog').open, true);
  h.context.deletedRequests = [];
  h.run(`request = async (path, options) => { deletedRequests.push([path, options.method]); return { deleted: true }; };`);
  await h.node('confirm-delete-session').onclick();
  assert.deepEqual(JSON.parse(JSON.stringify(h.context.deletedRequests)), [['/sessions/old-session', 'DELETE']]);
  assert.equal(h.node('delete-session-dialog').open, false);
});

test('deleting the open session removes its draft and starts a fresh empty session', async () => {
  const saved = new Map([['shiyu.draft.one', 'old draft'], ['shiyu.pendingDraft.one', 'pending']]);
  const h = harness(saved);
  h.run(`state.current = { id: 'one', character_id: 'companion', model_id: 'model',
    model_mode: 'base', messages: [] };
    state.characters = [{ id: 'companion', name: '小满', color: 'peach' }];
    state.health.model_modes = ['base'];
    state.pendingDeleteSessionId = 'one';`);
  h.context.calls = [];
  h.run(`request = async (path, options = {}) => {
    calls.push([path, options.method || 'GET']);
    if (path === '/sessions' && options.method === 'POST') return { id: 'fresh', character_id: 'companion', messages: [] };
    return { deleted: true };
  };`);
  await h.node('confirm-delete-session').onclick();
  assert.deepEqual(JSON.parse(JSON.stringify(h.context.calls)), [
    ['/sessions/one', 'DELETE'], ['/sessions', 'POST'],
  ]);
  assert.equal(h.run('state.current.id'), 'fresh');
  assert.equal(saved.has('shiyu.draft.one'), false);
  assert.equal(saved.has('shiyu.pendingDraft.one'), false);
});

test('retry recovery does not erase a newer draft', async () => {
  const h = harness(), pending = deferred();
  h.context.pending = pending.promise;
  h.run('request = () => pending');
  h.node('draft').value = '原来的问题';
  const sending = h.run('send(true)');
  h.node('draft').value = '修改后的新问题';
  h.node('draft').oninput();
  pending.resolve({ id: 'one', messages: [
    { role: 'user', content: '原来的问题' }, { role: 'assistant', content: '已保存的回答' },
  ] });
  await sending;
  assert.equal(h.node('draft').value, '修改后的新问题');
});

test('clearing history blocks navigation until the delete finishes', async () => {
  const h = harness(), pending = deferred();
  h.context.pending = pending.promise;
  h.run("request = path => path.endsWith('/messages') ? pending : Promise.resolve({})");
  const clearing = h.node('confirm-clear').onclick();
  assert.equal(h.run('canNavigate()'), false);
  pending.resolve({ id: 'one', messages: [] });
  await clearing;
  assert.equal(h.run('state.busy'), null);
});

test('failed delete releases busy state and leaves the session intact', async () => {
  const h = harness();
  h.run("request = async () => { throw new Error('模拟删除失败'); }");
  await h.node('confirm-clear').onclick();
  assert.equal(h.run('state.busy'), null);
  assert.equal(h.run('state.current.id'), 'one');
});

function waitForAbort(signal) {
  return new Promise((_, reject) => {
    const fail = () => reject(new Error('aborted by deadline'));
    if (signal.aborted) fail();
    else signal.addEventListener('abort', fail, { once: true });
  });
}

test('deadline aborts a request stuck before headers', async () => {
  const h = harness();
  let aborted = false;
  h.context.fetch = (_, options) => {
    options.signal.addEventListener('abort', () => { aborted = true; });
    return waitForAbort(options.signal);
  };
  await assert.rejects(h.run("originalRequest('/hung', {}, 5)"), /等待服务超时/);
  assert.equal(aborted, true);
});

test('deadline also covers a stalled JSON response body', async () => {
  const h = harness();
  h.context.fetch = async (_, options) => ({ ok: true,
    json: () => waitForAbort(options.signal) });
  await assert.rejects(h.run("originalRequest('/hung-body', {}, 5)"), /等待服务超时/);
});

test('successful response clears its deadline', async () => {
  const h = harness();
  let signal;
  h.context.fetch = async (_, options) => {
    signal = options.signal;
    return { ok: true, json: async () => ({ ok: true }) };
  };
  await h.run("originalRequest('/fast', {}, 5)");
  await new Promise(r => setTimeout(r, 15));
  assert.equal(signal.aborted, false);
});

test('timed out send preserves draft and releases the interface', async () => {
  const h = harness();
  h.run('withResponse = (path, options, consume) => originalWithResponse(path, options, consume, 5)');
  h.context.fetch = (_, options) => waitForAbort(options.signal);
  h.node('draft').value = '不要丢失这个问题';
  await h.run('send()');
  assert.equal(h.node('draft').value, '不要丢失这个问题');
  assert.equal(h.run('state.busy'), null);
  assert.match(h.node('chat-status').textContent, /超时/);
});

test('refresh removes unavailable LoRA and makes old-model history read only', async () => {
  const h = harness();
  h.node('model-mode').value = 'lora';
  h.node('draft').value = '保留草稿';
  h.context.fetch = async () => ({ ok: true, json: async () => ({
    status: 'ready', model_id: 'new-model', model_modes: ['base'],
  }) });
  await h.run('originalHealth()');
  assert.equal(h.node('model-mode').value, 'base');
  assert.equal(h.node('draft').readOnly, true);
  assert.equal(h.node('draft').value, '保留草稿');
});

test('send detects a changed backend model before submitting chat', async () => {
  const h = harness();
  h.run("refreshHealth = async () => (state.health = { status: 'ready', model_id: 'new-model' })");
  let calls = 0;
  h.context.fetch = () => { calls++; throw new Error('must not submit'); };
  h.node('draft').value = '保留草稿';
  await h.run('send()');
  assert.equal(calls, 0);
  assert.equal(h.node('draft').value, '保留草稿');
  assert.equal(h.run('state.busy'), null);
});

test('health can recover after an unavailable model becomes ready', async () => {
  const h = harness();
  let ready = false;
  h.context.fetch = async () => ({ ok: ready, status: ready ? 200 : 503,
    json: async () => ({ status: ready ? 'ready' : 'model_unavailable',
      model_id: ready ? 'model' : null, model_modes: ready ? ['base'] : [] }) });
  await h.run('originalHealth()');
  assert.equal(h.run('state.health.status'), 'model_unavailable');
  ready = true;
  await h.run('originalHealth()');
  assert.equal(h.run('state.health.status'), 'ready');
  assert.equal(h.node('draft').readOnly, false);
});

test('refresh while clearing cannot release the other operation lock', async () => {
  const h = harness();
  h.run("setBusy('navigation')");
  await h.node('refresh-history').onclick();
  assert.equal(h.run('state.busy'), 'navigation');
});

test('stalled streaming body also times out and restores the submitted question', async () => {
  const h = harness();
  h.run(`withResponse = (path, options, consume) => originalWithResponse(path, options, consume, 5);
    sse = originalSse;`);
  let released = false;
  h.context.fetch = async (_, options) => ({ ok: true, body: {
    getReader: () => ({ read: () => waitForAbort(options.signal),
      releaseLock: () => { released = true; } }),
  } });
  h.node('draft').value = '流式中断也请保留';
  await h.run('send()');
  assert.equal(h.node('draft').value, '流式中断也请保留');
  assert.equal(h.run('state.busy'), null);
  assert.equal(released, true);
});

test('a refresh failure preserves the current session and draft', async () => {
  const h = harness();
  h.context.fetch = async () => { throw new Error('模拟服务断开'); };
  h.node('draft').value = '服务恢复后接着发';
  await assert.rejects(h.run('originalHealth()'), /模拟服务断开/);
  assert.equal(h.run('state.health.status'), 'disconnected');
  assert.equal(h.run('state.current.id'), 'one');
  assert.equal(h.node('draft').value, '服务恢复后接着发');
});

test('history pagination appends unique rows and counts raw page offsets', async () => {
  const h = harness();
  h.run('renderHistory = () => {}');
  const calls = [];
  h.context.pageRequest = async url => {
    calls.push(url);
    return calls.length === 1
      ? { sessions: [{ id: 'a' }, { id: 'b' }], has_more: true }
      : { sessions: [{ id: 'b' }, { id: 'c' }], has_more: false };
  };
  h.run('request = pageRequest');
  await h.run('loadHistory()');
  await h.node('history-more').onclick();
  assert.deepEqual(Array.from(h.run('state.historySessions.map(r => r.id)')), ['a', 'b', 'c']);
  assert.match(calls[1], /offset=2/);
  assert.equal(h.node('history-more').hidden, true);
});

test('older search response cannot replace newer search results', async () => {
  const h = harness(), first = deferred();
  h.run('renderHistory = () => {}');
  let calls = 0;
  h.context.pageRequest = () => ++calls === 1 ? first.promise
    : Promise.resolve({ sessions: [{ id: 'new' }], has_more: false });
  h.run('request = pageRequest');
  h.node('history-search').value = 'old';
  const old = h.run('loadHistory()');
  h.node('history-search').value = 'new';
  await h.run('loadHistory()');
  first.resolve({ sessions: [{ id: 'old' }], has_more: true });
  await old;
  assert.equal(h.run('state.historySessions[0].id'), 'new');
  assert.equal(h.node('history-more').hidden, true);
});

test('typing invalidates a pending history response before debounce completes', async () => {
  const h = harness(), pending = deferred();
  h.run('renderHistory = () => {}');
  h.context.pending = pending.promise;
  h.run('request = () => pending');
  const old = h.run('loadHistory()');
  h.node('history-search').value = '新关键词';
  h.node('history-search').oninput();
  h.run('clearTimeout(historySearchTimer)');
  pending.resolve({ sessions: [{ id: 'stale' }], has_more: true });
  await old;
  assert.equal(h.run('state.historySessions.length'), 0);
  assert.equal(h.node('history-more').hidden, true);
});

test('failed load more preserves rows and retries at the same offset', async () => {
  const h = harness();
  h.run(`renderHistory = () => {};
    state.historySessions = [{ id: 'saved' }];
    state.historyOffset = 20; state.historyHasMore = true;
    request = async () => { throw new Error('模拟断网'); };`);
  await h.run('loadHistory(true)');
  assert.equal(h.run('state.historyOffset'), 20);
  assert.equal(h.run('state.historySessions[0].id'), 'saved');
  assert.equal(h.node('history-more').disabled, false);
  assert.match(h.node('history-status').textContent, /模拟断网/);
});

function microphone(h, getUserMedia) {
  const recorders = [];
  h.context.navigator = { mediaDevices: { getUserMedia } };
  h.context.MediaRecorder = class {
    static isTypeSupported() { return true; }
    constructor(stream) { this.stream = stream; this.state = 'inactive'; this.mimeType = 'audio/webm'; recorders.push(this); }
    start() { this.state = 'recording'; }
    stop() { this.state = 'inactive'; this.onstop?.(); }
  };
  h.node('voice-dialog').open = true;
  return recorders;
}

test('pending microphone permission blocks duplicate start and navigation', async () => {
  const h = harness(), pending = deferred();
  let calls = 0, stops = 0;
  microphone(h, () => { calls++; return pending.promise; });
  const recording = h.run('record()');
  await h.run('record()');
  assert.equal(calls, 1);
  assert.equal(h.run('canNavigate()'), false);
  h.run('releaseRecorder(true)');
  pending.resolve({ getTracks: () => [{ stop() { stops++; } }] });
  await recording;
  assert.equal(stops, 1);
  assert.equal(h.run('state.recording'), null);
});

test('late rejected permission cannot stop a newer recording', async () => {
  const h = harness();
  let rejectOld, calls = 0;
  const old = new Promise((_, reject) => { rejectOld = reject; });
  microphone(h, () => ++calls === 1 ? old : Promise.resolve({ getTracks: () => [{ stop() {} }] }));
  const first = h.run('record()');
  h.run('releaseRecorder(true)');
  await h.run('record()');
  rejectOld(new Error('old permission failed'));
  await first;
  assert.equal(h.run('state.recording.state'), 'recording');
  h.run('releaseRecorder(true)');
});

test('device-driven stop releases tracks and timer while preserving the audio', async () => {
  const h = harness();
  let stops = 0;
  const recorders = microphone(h, async () => ({ getTracks: () => [{ stop() { stops++; } }] }));
  await h.run('record()');
  recorders[0].ondataavailable({ data: new Blob(['fake audio']) });
  recorders[0].state = 'inactive';
  recorders[0].onstop();
  assert.equal(h.run('state.recordTimer'), null);
  assert.equal(h.run('state.recording'), null);
  assert.ok(h.run('state.voiceFile.size') > 0);
  assert.ok(stops > 0);
});

test('recorder error releases microphone and allows retry', async () => {
  const h = harness();
  const recorders = microphone(h, async () => ({ getTracks: () => [{ stop() {} }] }));
  await h.run('record()');
  recorders[0].onerror();
  assert.equal(h.run('state.recording'), null);
  assert.equal(h.run('state.recordTimer'), null);
  assert.equal(h.node('record-audio').disabled, false);
  assert.match(h.node('voice-status').textContent, /设备发生错误/);
});

test('new recording cannot transcribe leftover audio while permission is pending', async () => {
  const h = harness(), pending = deferred();
  microphone(h, () => pending.promise);
  h.run('state.voiceFile = new Blob(["old recording"])');
  let requests = 0;
  h.context.countRequest = () => { requests++; };
  h.run('request = countRequest');
  const recording = h.run('record()');
  await h.run('transcribe()');
  assert.equal(requests, 0);
  assert.equal(h.node('transcribe-audio').disabled, true);
  h.run('releaseRecorder(true)');
  pending.resolve({ getTracks: () => [{ stop() {} }] });
  await recording;
});

test('ASR completion cannot overwrite a newer text draft', async () => {
  const h = harness(), pending = deferred();
  h.context.pending = pending.promise;
  h.run('state.voiceFile = new Blob(["fake recording"]); request = () => pending');
  const recognition = h.run('transcribe()');
  h.node('draft').value = '我正在自己输入';
  h.node('draft').oninput();
  pending.resolve({ text: '较晚返回的识别文字' });
  await recognition;
  assert.equal(h.node('draft').value, '我正在自己输入');
  assert.match(h.node('voice-status').textContent, /较晚返回的识别文字/);
  assert.equal(h.run('state.busy'), null);
});

test('closing voice input invalidates an outstanding transcription', async () => {
  const h = harness(), pending = deferred();
  h.context.pending = pending.promise;
  h.run('state.voiceFile = new Blob(["fake recording"]); request = () => pending');
  h.node('draft').value = '保留这段草稿';
  const recognition = h.run('transcribe()');
  h.run('releaseRecorder(true)');
  pending.resolve({ text: '关闭以后返回的文字' });
  await recognition;
  assert.equal(h.node('draft').value, '保留这段草稿');
  assert.equal(h.run('state.busy'), null);
});

function enableActivation(h) {
  h.run(`avatar = () => {}; renderProfile = () => {}; showView = () => {};
    renderSessions = () => {};
    state.characters = [{id:'assistant',name:'test',tag:'test'}];`);
}

test('new frontend instance restores an unsent draft from this browser', () => {
  const first = harness();
  first.node('draft').value = '刷新也要保留的草稿';
  first.node('draft').oninput();
  const second = harness(first.savedStorage);
  enableActivation(second);
  second.run("state.current = null; originalActivate({id:'one',messages:[],character_id:'assistant'})");
  assert.equal(second.node('draft').value, '刷新也要保留的草稿');
});

test('switching sessions saves and restores separate drafts', () => {
  const h = harness();
  enableActivation(h);
  h.node('draft').value = '第一段草稿';
  h.run("originalActivate({id:'two',messages:[],character_id:'assistant'})");
  h.node('draft').value = '第二段草稿';
  h.run("originalActivate({id:'one',messages:[],character_id:'assistant'})");
  assert.equal(h.node('draft').value, '第一段草稿');
  h.run("originalActivate({id:'two',messages:[],character_id:'assistant'})");
  assert.equal(h.node('draft').value, '第二段草稿');
});

test('clear confirmation clears both draft and pending question', () => {
  const h = harness();
  enableActivation(h);
  h.run("storage.set('draft.one','草稿'); storage.set('pendingDraft.one','待确认的问题')");
  h.run("originalActivate({id:'one',messages:[],character_id:'assistant'},true)");
  assert.equal(h.node('draft').value, '');
  assert.equal(h.run("storage.get('pendingDraft.one','')"), '');
});

test('reload during an unfinished stream restores the submitted question', async () => {
  const first = harness(), entered = deferred(), finish = deferred();
  first.context.entered = entered;
  first.context.finish = finish.promise;
  first.context.fetch = async () => ({ok:true});
  first.run("sse = async function*(){ entered.resolve(); await finish; yield {type:'done',data:{metrics:{}}}; }");
  first.node('draft').value = '流式未结束的问题';
  const sending = first.run('send()');
  await entered.promise;
  assert.equal(first.node('draft').value, '');
  const second = harness(first.savedStorage);
  enableActivation(second);
  second.run("state.current=null; originalActivate({id:'one',messages:[],character_id:'assistant'})");
  assert.equal(second.node('draft').value, '流式未结束的问题');
  assert.equal(second.node('retry-message').hidden, false);
  finish.resolve();
  await sending;
});

test('saved answer after a lost connection prevents restoring an already completed question', () => {
  const h = harness();
  enableActivation(h);
  h.run(`state.current=null; storage.set('pendingDraft.one','问题');
    originalActivate({id:'one',character_id:'assistant',messages:[
      {role:'user',content:'问题'},{role:'assistant',content:'已保存的回答'}]})`);
  assert.equal(h.node('draft').value, '');
  assert.equal(h.run("storage.get('pendingDraft.one','')"), '');
  assert.equal(h.node('retry-message').hidden, true);
});

test('blocked browser storage keeps the visible draft and explains persistence failure', () => {
  const h = harness();
  h.context.localStorage.setItem = () => { throw new Error('storage disabled'); };
  h.node('draft').value = '不要丢失';
  h.node('draft').oninput();
  assert.equal(h.node('draft').value, '不要丢失');
  assert.match(h.node('toast').textContent, /无法保存草稿/);
});
