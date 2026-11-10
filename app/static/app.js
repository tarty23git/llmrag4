"use strict";

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const questionEl = $("question");
const sendBtn = $("send");
let history = [];      // [{role, content}] sent back to the server for follow-ups
let busy = false;

// ---------- minimal, escape-first Markdown renderer ----------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\[(\d{1,3})\]/g, '<button class="cite" data-n="$1">$1</button>');
}

function renderMarkdown(src) {
  const blocks = [];
  // Pull out fenced code first so nothing inside it is transformed.
  src = src.replace(/```[\w-]*\n?([\s\S]*?)(```|$)/g, (_, code) => {
    blocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of escapeHtml(src).split("\n")) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^\u0000(\d+)\u0000$/))) { closeList(); out.push(blocks[+m[1]]); }
    else if ((m = line.match(/^#{1,6}\s+(.*)/))) { closeList(); out.push(`<h4>${inline(m[1])}</h4>`); }
    else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if (!line.trim()) { closeList(); }
    else { closeList(); out.push(`<p>${inline(line)}</p>`); }
  }
  closeList();
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}

// ---------- status & documents ----------
async function refreshStatus() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("status").textContent = `${h.model} · ${h.retrieval}`;
    $("keyBanner").hidden = !!h.api_key_configured;
  } catch { $("status").textContent = "Server unreachable"; }
}

async function refreshDocs() {
  const docs = await (await fetch("/api/documents")).json();
  $("docCount").textContent = docs.length ? `${docs.length}` : "";
  const list = $("docList");
  list.innerHTML = "";
  if (!docs.length) {
    list.innerHTML = '<li class="muted">No documents yet.</li>';
    return;
  }
  for (const d of docs) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name"></span><span class="muted">${d.chunks}</span><button aria-label="Remove">✕</button>`;
    li.querySelector(".name").textContent = d.source;
    li.querySelector(".name").title = `${d.source} · ${d.chunks} chunks`;
    li.querySelector("button").onclick = async () => {
      if (!confirm(`Remove "${d.source}" from the index?`)) return;
      await fetch(`/api/documents/${d.doc_id}`, { method: "DELETE" });
      refreshDocs(); refreshStatus();
    };
    list.appendChild(li);
  }
}

async function upload(files) {
  if (!files.length) return;
  const log = $("uploadLog");
  log.innerHTML = `<div class="muted">Indexing ${files.length} file(s)…</div>`;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  try {
    const res = await fetch("/api/documents", { method: "POST", body: form });
    const results = await res.json();
    log.innerHTML = "";
    for (const r of (Array.isArray(results) ? results : [])) {
      const div = document.createElement("div");
      if (r.error) { div.className = "err"; div.textContent = `✗ ${r.source}: ${r.error}`; }
      else div.textContent = r.skipped ? `= ${r.source} (already indexed)` : `✓ ${r.source}: ${r.chunks} chunks`;
      log.appendChild(div);
    }
  } catch (e) {
    log.innerHTML = `<div class="err">Upload failed: ${escapeHtml(String(e))}</div>`;
  }
  refreshDocs(); refreshStatus();
}

const dz = $("dropzone");
$("fileInput").onchange = (e) => { upload([...e.target.files]); e.target.value = ""; };
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("drag"); };
dz.ondragleave = () => dz.classList.remove("drag");
dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("drag"); upload([...e.dataTransfer.files]); };

// ---------- chat ----------
function addMessage(role) {
  $("empty")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = '<div class="bubble"></div>';
  messagesEl.appendChild(wrap);
  return wrap;
}

function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

function showSource(src) {
  $("sourceTitle").textContent = `[${src.n}] ${src.source} · chunk ${src.chunk}`;
  $("sourceText").textContent = src.text;
  $("sourceDialog").showModal();
}

async function ask(question) {
  busy = true; sendBtn.disabled = true;
  addMessage("user").querySelector(".bubble").textContent = question;
  const msg = addMessage("assistant");
  const bubble = msg.querySelector(".bubble");
  bubble.classList.add("typing");
  scrollToBottom();

  let text = "", sources = [];
  const render = () => { bubble.innerHTML = renderMarkdown(text); };
  bubble.onclick = (e) => {
    const n = e.target.closest(".cite")?.dataset.n;
    const src = n && sources.find((s) => s.n === +n);
    if (src) showSource(src);
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "sources") sources = ev.sources;
        else if (ev.type === "text") { text += ev.text; render(); scrollToBottom(); }
        else if (ev.type === "error") {
          const p = document.createElement("div"); p.className = "notice error"; p.textContent = ev.message; msg.appendChild(p);
          if (ev.code === "missing_api_key" || ev.code === "invalid_api_key") {
            const btn = document.createElement("button"); btn.className = "text-btn"; btn.textContent = "Add API key";
            btn.style.marginTop = "8px"; btn.onclick = openKeyDialog; msg.appendChild(btn);
            $("keyBanner").hidden = false;
          }
        } else if (ev.type === "done" && ev.notice) {
          const p = document.createElement("div"); p.className = "notice"; p.textContent = ev.notice; msg.appendChild(p);
        }
      }
    }
  } catch (e) {
    const p = document.createElement("div"); p.className = "notice error"; p.textContent = `Request failed: ${e.message}`; msg.appendChild(p);
  }

  bubble.classList.remove("typing");
  render();
  if (sources.length) {
    const used = new Set([...text.matchAll(/\[(\d{1,3})\]/g)].map((m) => +m[1]));
    const shown = used.size ? sources.filter((s) => used.has(s.n)) : sources;
    const row = document.createElement("div");
    row.className = "sources";
    for (const s of shown) {
      const b = document.createElement("button");
      b.className = "source-chip";
      b.textContent = `[${s.n}] ${s.source.split("/").pop()} · ${s.chunk}`;
      b.onclick = () => showSource(s);
      row.appendChild(b);
    }
    msg.appendChild(row);
  }
  if (text) history.push({ role: "user", content: question }, { role: "assistant", content: text });
  history = history.slice(-20);
  scrollToBottom();
  busy = false; sendBtn.disabled = false; questionEl.focus();
}

$("composer").onsubmit = (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (!q || busy) return;
  questionEl.value = ""; questionEl.style.height = "auto";
  ask(q);
};
questionEl.onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
};
questionEl.oninput = () => { questionEl.style.height = "auto"; questionEl.style.height = `${questionEl.scrollHeight}px`; };
$("newChat").onclick = () => {
  history = [];
  messagesEl.innerHTML = '<div class="empty" id="empty"><h2>What would you like to know?</h2><p class="muted">Ask a question about your documents.</p></div>';
};
$("toggleSidebar").onclick = (e) => { e.stopPropagation(); $("sidebar").classList.toggle("open"); };
// On narrow screens the drawer covers the toggle; tapping the chat area closes it.
document.querySelector(".chat").addEventListener("click", () => $("sidebar").classList.remove("open"));

// ---------- API key ----------
function openKeyDialog() {
  $("keyMsg").textContent = ""; $("keyMsg").className = "key-msg";
  $("keyInput").value = "";
  $("keyDialog").showModal();
  $("keyInput").focus();
}
$("keyBtn").onclick = openKeyDialog;
$("keyBannerBtn").onclick = openKeyDialog;
$("keyClose").onclick = () => $("keyDialog").close();
$("keyForm").onsubmit = async (e) => {
  e.preventDefault();
  const key = $("keyInput").value.trim();
  if (!key) return;
  const msg = $("keyMsg"), btn = $("keySave");
  btn.disabled = true; msg.className = "key-msg"; msg.textContent = "Checking the key with Anthropic…";
  try {
    const res = await fetch("/api/settings/api-key", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: key }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = data.detail;
      throw new Error(typeof d === "string" ? d : "That doesn't look like a valid key.");
    }
    msg.className = "key-msg ok"; msg.textContent = "Key verified and saved. You can ask questions now.";
    $("keyInput").value = "";
    refreshStatus();
    setTimeout(() => { $("keyDialog").close(); questionEl.focus(); }, 900);
  } catch (err) {
    msg.className = "key-msg err"; msg.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
};

refreshStatus();
refreshDocs();
