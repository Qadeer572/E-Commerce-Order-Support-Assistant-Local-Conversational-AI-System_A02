/**
 * ShopBot — Enhanced Frontend Application
 *
 * Handles:
 *  - Animated canvas background (floating particles + mouse parallax)
 *  - WebSocket connection management with auto-reconnect
 *  - Streaming token display (word-by-word as received)
 *  - Session creation and reset via REST API
 *  - Intent badge updates + capability highlights
 *  - Markdown-like rendering (bold, lists, tables)
 *  - Error toasts and connection status
 */

'use strict';

// ── Animated Canvas Background ────────────────────────────────────────────────
(function initCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, particles, mouse = { x: 0, y: 0 };

  const PARTICLE_COUNT = 55;
  const COLORS = ['#5b9cf6', '#7b61ff', '#22d3ee', '#4ade80', '#c084fc'];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle() {
    return {
      x:   Math.random() * W,
      y:   Math.random() * H,
      r:   Math.random() * 1.8 + 0.4,
      vx:  (Math.random() - 0.5) * 0.35,
      vy:  (Math.random() - 0.5) * 0.35,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: Math.random() * 0.5 + 0.15,
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.012 + Math.random() * 0.018,
    };
  }

  function init() {
    resize();
    particles = Array.from({ length: PARTICLE_COUNT }, mkParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const p = particles[i], q = particles[j];
        const dx = p.x - q.x, dy = p.y - q.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.strokeStyle = p.color;
          ctx.globalAlpha = (1 - dist / 130) * 0.06;
          ctx.lineWidth = 0.5;
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(q.x, q.y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    for (const p of particles) {
      p.pulse += p.pulseSpeed;
      const pulse = 0.7 + Math.sin(p.pulse) * 0.3;

      // Mouse parallax repulsion
      const dx = p.x - mouse.x, dy = p.y - mouse.y;
      const d  = Math.sqrt(dx * dx + dy * dy);
      if (d < 100) {
        p.x += (dx / d) * 0.6;
        p.y += (dy / d) * 0.6;
      }

      ctx.beginPath();
      ctx.globalAlpha = p.alpha * pulse;
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.arc(p.x, p.y, p.r * pulse, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Move
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -10) p.x = W + 10;
      if (p.x > W + 10) p.x = -10;
      if (p.y < -10) p.y = H + 10;
      if (p.y > H + 10) p.y = -10;
    }

    ctx.globalAlpha = 1;
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  init();
  draw();
})();

// ── Configuration ────────────────────────────────────────────────────────────
const CONFIG = {
  WS_URL:          `ws://${location.host}/ws/chat`,
  API_BASE:        `${location.protocol}//${location.host}/api`,
  RECONNECT_DELAY: 3000,   // ms
  MAX_RECONNECTS:  5,
  SESSION_KEY:     'shopbot_session_id',
};

// ── State ────────────────────────────────────────────────────────────────────
let ws              = null;
let sessionId       = null;
let isStreaming     = false;
let reconnectCount  = 0;
let reconnectTimer  = null;
let currentIntent   = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $messagesList    = document.getElementById('messages-list');
const $messagesContainer = document.getElementById('messages-container');
const $messageInput    = document.getElementById('message-input');
const $sendBtn         = document.getElementById('send-btn');
const $typingIndicator = document.getElementById('typing-indicator');
const $intentBadge     = document.getElementById('intent-badge');
const $statusDot       = document.getElementById('status-dot');
const $statusText      = document.getElementById('status-text');
const $sessionInfo     = document.getElementById('session-info');
const $charCount       = document.getElementById('char-count');
const $toast           = document.getElementById('toast');
const $newSessionBtn   = document.getElementById('new-session-btn');
const $sidebarToggle   = document.getElementById('sidebar-toggle');
const $sidebar         = document.getElementById('sidebar');
const $suggestionBtns  = document.querySelectorAll('.suggestion-btn');
const $capProduct      = document.getElementById('cap-product');
const $capOrder        = document.getElementById('cap-order');
const $capPolicy       = document.getElementById('cap-policy');
const $historyList     = document.getElementById('history-list');

// ── Voice DOM refs (Assignment 3) ───────────────────────────────────────────
const $voiceBtn         = document.getElementById('voice-btn');
const $voiceBar         = document.getElementById('voice-bar');
const $voiceStateText   = document.getElementById('voice-state-text');
const $voiceTranscriptPreview = document.getElementById('voice-transcript-preview');
const $voiceStopBtn     = document.getElementById('voice-stop-btn');
const $voiceDoneBtn     = document.getElementById('voice-done-btn');
const $voiceCancelBtn   = document.getElementById('voice-cancel-btn');
const $headerVoicePill  = document.getElementById('header-voice-pill');

// ── Session management ────────────────────────────────────────────────────────
function loadOrCreateSessionId() {
  let id = sessionStorage.getItem(CONFIG.SESSION_KEY);
  if (!id) {
    id = generateUUID();
    sessionStorage.setItem(CONFIG.SESSION_KEY, id);
  }
  return id;
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function resetSession() {
  if (isStreaming) {
    showToast('Please wait for the current response to finish.', 'error');
    return;
  }

  // Generate a new session ID. The previous conversation is left untouched
  // on the server so it keeps showing up in the History sidebar.
  sessionId = generateUUID();
  sessionStorage.setItem(CONFIG.SESSION_KEY, sessionId);
  currentIntent = null;

  // Clear chat UI
  $messagesList.innerHTML = '';
  updateIntentBadge(null);
  updateCapabilityHighlight(null);

  // Re-add welcome message
  appendWelcomeMessage();

  showToast('New conversation started ✨', 'success');
  $sessionInfo.textContent = `Session: ${sessionId.slice(0, 8)}…`;
  renderActiveHistoryItem();
}

// ── Conversation history sidebar ──────────────────────────────────────────────
function timeAgo(epochSeconds) {
  const diffMs = Date.now() - epochSeconds * 1000;
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

async function loadHistoryList() {
  try {
    const res = await fetch(`${CONFIG.API_BASE}/sessions`);
    if (!res.ok) return;
    const data = await res.json();
    renderHistoryList(data.sessions || []);
  } catch {
    /* history sidebar is a non-critical enhancement — fail silently */
  }
}

function renderHistoryList(sessions) {
  $historyList.innerHTML = '';

  if (!sessions.length) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    empty.id = 'history-empty';
    empty.textContent = 'No past conversations yet';
    $historyList.appendChild(empty);
    return;
  }

  for (const s of sessions) {
    const item = document.createElement('button');
    item.className = 'history-item' + (s.session_id === sessionId ? ' active' : '');
    item.dataset.sessionId = s.session_id;

    const body = document.createElement('div');
    body.className = 'history-item-body';

    const title = document.createElement('span');
    title.className = 'history-item-title';
    title.textContent = s.title;

    const meta = document.createElement('span');
    meta.className = 'history-item-meta';
    meta.textContent = `${s.message_count} msg${s.message_count === 1 ? '' : 's'} · ${timeAgo(s.last_active)}`;

    body.appendChild(title);
    body.appendChild(meta);

    const del = document.createElement('span');
    del.className = 'history-item-delete';
    del.textContent = '🗑️';
    del.title = 'Delete conversation';
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteHistorySession(s.session_id);
    });

    item.appendChild(body);
    item.appendChild(del);

    item.addEventListener('click', () => selectHistorySession(s.session_id));

    $historyList.appendChild(item);
  }
}

function renderActiveHistoryItem() {
  document.querySelectorAll('.history-item').forEach(el => {
    el.classList.toggle('active', el.dataset.sessionId === sessionId);
  });
}

async function selectHistorySession(id) {
  if (id === sessionId) return;
  if (isStreaming) {
    showToast('Please wait for the current response to finish.', 'error');
    return;
  }

  try {
    const res = await fetch(`${CONFIG.API_BASE}/session/${id}/history`);
    if (!res.ok) {
      showToast('Could not load that conversation.', 'error');
      return;
    }
    const data = await res.json();

    sessionId = id;
    sessionStorage.setItem(CONFIG.SESSION_KEY, sessionId);
    currentIntent = null;

    $messagesList.innerHTML = '';
    if (!data.messages.length) {
      appendWelcomeMessage();
    } else {
      for (const m of data.messages) {
        if (m.role === 'user') appendUserMessage(m.content);
        else appendBotMessage(m.content);
      }
      // Re-derive the last known intent from the latest user message
      const lastUserMsg = [...data.messages].reverse().find(m => m.role === 'user');
      if (lastUserMsg) {
        const intent = detectIntentFromText(lastUserMsg.content);
        if (intent) updateIntentBadge(intent);
      }
    }

    $sessionInfo.textContent = `Session: ${sessionId.slice(0, 8)}…`;
    renderActiveHistoryItem();
    scrollToBottom();

    if (window.innerWidth < 640) $sidebar.classList.remove('open');
  } catch {
    showToast('Could not load that conversation.', 'error');
  }
}

async function deleteHistorySession(id) {
  try {
    await fetch(`${CONFIG.API_BASE}/history/${id}`, { method: 'DELETE' });
  } catch { /* ignore */ }

  if (id === sessionId) {
    resetSession();
  }
  loadHistoryList();
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  setStatus('connecting');
  ws = new WebSocket(CONFIG.WS_URL);

  ws.onopen = () => {
    setStatus('online');
    reconnectCount = 0;
    clearTimeout(reconnectTimer);
    enableInput(true);
  };

  ws.onclose = (event) => {
    setStatus('offline');
    enableInput(false);

    if (isStreaming) {
      finishStreaming(true /* disconnected */);
    }

    if (reconnectCount < CONFIG.MAX_RECONNECTS) {
      reconnectCount++;
      const delay = CONFIG.RECONNECT_DELAY * reconnectCount;
      setStatus('reconnecting', reconnectCount);
      reconnectTimer = setTimeout(connect, delay);
    } else {
      showToast('Connection lost. Please refresh the page.', 'error');
    }
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };

  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      console.error('Received non-JSON WS message:', event.data);
      return;
    }
    handleServerMessage(data);
  };
}

// ── Incoming message handler ──────────────────────────────────────────────────
let $currentBotBubble = null;
let currentBotText    = '';

function handleServerMessage(data) {
  const { type, content } = data;

  if (type === 'token') {
    if (!isStreaming) {
      // First token — start a new bot message
      isStreaming = true;
      hideTypingIndicator();
      setStatus('typing');
      $currentBotBubble = appendBotMessage('', /* streaming */ true);
      currentBotText = '';
    }
    currentBotText += content;
    renderStreamingContent($currentBotBubble, currentBotText);
    scrollToBottom();

  } else if (type === 'done') {
    finishStreaming(false);

  } else if (type === 'error') {
    finishStreaming(false);
    if ($currentBotBubble) {
      $currentBotBubble.classList.add('error-bubble');
      $currentBotBubble.innerHTML = `⚠️ ${escapeHtml(content)}`;
    } else {
      appendErrorMessage(content);
    }
    showToast(content, 'error');
  }
}

function finishStreaming(disconnected = false) {
  isStreaming = false;
  setStatus('online');
  hideTypingIndicator();
  enableInput(true);

  if ($currentBotBubble) {
    // Remove streaming cursor class
    $currentBotBubble.classList.remove('streaming-cursor');

    if (!disconnected && currentBotText) {
      // Detect intent from the response and update badge
      const detectedIntent = detectIntentFromResponse(currentBotText);
      if (detectedIntent) updateIntentBadge(detectedIntent);
    }
  }

  $currentBotBubble = null;
  currentBotText    = '';
  scrollToBottom();

  if (!disconnected) loadHistoryList();
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage(text) {
  if (!text.trim() || isStreaming) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    showToast('Not connected. Reconnecting…', 'error');
    connect();
    return;
  }

  // Append user bubble
  appendUserMessage(text);

  // Detect intent client-side for immediate badge/highlight feedback
  const clientIntent = detectIntentFromText(text);
  if (clientIntent) {
    updateIntentBadge(clientIntent);
    updateCapabilityHighlight(clientIntent);
  }

  // Show typing indicator
  showTypingIndicator();
  enableInput(false);

  // Send over WebSocket
  ws.send(JSON.stringify({ session_id: sessionId, message: text }));
  scrollToBottom();
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
let currentToolChip = null;

function appendToolChip(name, args) {
  const icon = name === 'calculator' ? '🧮' : (name === 'catalog_query' ? '📦' : (name === 'support_calendar' ? '📅' : '👤'));
  const chip = document.createElement('div');
  chip.className = 'tool-chip tool-running';
  chip.innerHTML = `<span>${icon}</span> Using <span class="tool-chip-name">${escapeHtml(name)}</span>…`;
  $messagesList.appendChild(chip);
  currentToolChip = chip;
  scrollToBottom();
}

function updateToolChip(name, ok, elapsedMs) {
  if (currentToolChip) {
    currentToolChip.classList.remove('tool-running');
    const icon = ok ? '✓' : '⚠️';
    const status = ok ? 'Completed' : 'Failed';
    currentToolChip.innerHTML = `<span>${icon}</span> Tool <span class="tool-chip-name">${escapeHtml(name)}</span> ${status} <span class="tool-chip-time">(${elapsedMs}ms)</span>`;
    currentToolChip = null;
  }
}

function appendMetricsChip(bubble, metrics) {
  if (!metrics || !bubble) return;
  const chip = document.createElement('div');
  chip.className = 'metrics-chip';
  const parts = [];
  if (metrics.asr_finalize_ms) parts.push(`ASR: ${metrics.asr_finalize_ms}ms`);
  if (metrics.tool_ms) parts.push(`Tool: ${metrics.tool_ms}ms`);
  if (metrics.llm_first_token_ms) parts.push(`LLM: ${Math.round(metrics.llm_first_token_ms)}ms`);
  if (metrics.tts_first_audio_ms) parts.push(`TTS: ${Math.round(metrics.tts_first_audio_ms)}ms`);
  if (metrics.end_of_speech_to_first_audio_ms) parts.push(`Latency: ${Math.round(metrics.end_of_speech_to_first_audio_ms)}ms`);
  chip.textContent = `⚡ ` + parts.join(' · ');
  bubble.parentElement.appendChild(chip);
}

function appendUserMessage(text, isVoice = false) {
  const div = document.createElement('div');
  div.className = 'message message-user';
  const voiceTag = isVoice ? '<span class="voice-tag">🎙️ Voice</span>' : '';
  div.innerHTML = `
    <div class="message-avatar">
      <span>👤</span>
    </div>
    <div class="message-content">
      <div class="message-bubble">${escapeHtml(text)}</div>
      <div class="message-time">You ${voiceTag}</div>
    </div>`;
  $messagesList.appendChild(div);
}

function appendBotMessage(text, streaming = false) {
  const div = document.createElement('div');
  div.className = 'message message-assistant';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble' + (streaming ? ' streaming-cursor' : '');

  if (text) {
    bubble.innerHTML = renderMarkdown(text);
  }

  const avatarDiv = document.createElement('div');
  avatarDiv.className = 'message-avatar bot-avatar';
  avatarDiv.innerHTML = '<span>🤖</span>';
  div.appendChild(avatarDiv);

  const content = document.createElement('div');
  content.className = 'message-content';
  content.appendChild(bubble);

  const time = document.createElement('div');
  time.className = 'message-time';
  time.textContent = 'ShopBot';
  content.appendChild(time);

  div.appendChild(content);
  $messagesList.appendChild(div);
  return bubble;
}

function appendErrorMessage(text) {
  const div = document.createElement('div');
  div.className = 'message message-assistant message-error';
  div.innerHTML = `
    <div class="message-avatar bot-avatar"><span>🤖</span></div>
    <div class="message-content">
      <div class="message-bubble">⚠️ ${escapeHtml(text)}</div>
      <div class="message-time">ShopBot</div>
    </div>`;
  $messagesList.appendChild(div);
}

function appendWelcomeMessage() {
  const div = document.createElement('div');
  div.className = 'message message-assistant';
  div.innerHTML = `
    <div class="message-avatar bot-avatar"><span>🤖</span></div>
    <div class="message-content">
      <div class="message-bubble">
        <p>👋 Hi! I'm <strong>ShopBot</strong>, your order support assistant.</p>
        <p>I can help you with:</p>
        <ul>
          <li>🛍️ <strong>Product questions</strong> — specs, pricing, availability</li>
          <li>📦 <strong>Order tracking</strong> — status, shipment, delivery</li>
          <li>🔄 <strong>Returns &amp; policy</strong> — refunds, exchanges, shipping</li>
        </ul>
        <p>What can I help you with today?</p>
      </div>
      <div class="message-time">ShopBot</div>
    </div>`;
  $messagesList.appendChild(div);
}

function renderStreamingContent(bubble, text) {
  bubble.innerHTML = renderMarkdown(text);
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function showTypingIndicator() {
  $typingIndicator.style.display = 'flex';
  scrollToBottom();
}

function hideTypingIndicator() {
  $typingIndicator.style.display = 'none';
}

// ── Scroll ────────────────────────────────────────────────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    $messagesContainer.scrollTop = $messagesContainer.scrollHeight;
  });
}

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(state, count = 0) {
  $statusDot.className = 'status-dot';
  switch (state) {
    case 'online':
      $statusDot.classList.add('online');
      $statusText.textContent = 'Online';
      break;
    case 'offline':
      $statusDot.classList.add('offline');
      $statusText.textContent = 'Disconnected';
      break;
    case 'connecting':
      $statusText.textContent = 'Connecting…';
      break;
    case 'reconnecting':
      $statusText.textContent = `Reconnecting (${count}/${CONFIG.MAX_RECONNECTS})…`;
      break;
    case 'typing':
      $statusDot.classList.add('typing');
      $statusText.textContent = 'ShopBot is typing…';
      break;
  }
}

// ── Input state ───────────────────────────────────────────────────────────────
function enableInput(enabled) {
  $messageInput.disabled = !enabled;
  $sendBtn.disabled = !enabled || !$messageInput.value.trim();
  if (enabled) $messageInput.focus();
}

// ── Intent badge ──────────────────────────────────────────────────────────────
const INTENT_CONFIG = {
  PRODUCT_QUERY:  { label: '🛍️ Product', cls: 'product', cap: 'product' },
  ORDER_TRACKING: { label: '📦 Order',   cls: 'order',   cap: 'order'   },
  RETURNS_POLICY: { label: '🔄 Policy',  cls: 'policy',  cap: 'policy'  },
};

function updateIntentBadge(intent) {
  if (!intent || !INTENT_CONFIG[intent]) {
    $intentBadge.style.display = 'none';
    $intentBadge.className = 'header-intent-badge';
    return;
  }
  const cfg = INTENT_CONFIG[intent];
  $intentBadge.textContent = cfg.label;
  $intentBadge.className = `header-intent-badge ${cfg.cls}`;
  $intentBadge.style.display = 'block';
  currentIntent = intent;
  updateCapabilityHighlight(intent);
}

function updateCapabilityHighlight(intent) {
  [$capProduct, $capOrder, $capPolicy].forEach(el => el.classList.remove('active'));
  if (!intent) return;
  if (intent === 'PRODUCT_QUERY')  $capProduct.classList.add('active');
  if (intent === 'ORDER_TRACKING') $capOrder.classList.add('active');
  if (intent === 'RETURNS_POLICY') $capPolicy.classList.add('active');
}

// ── Client-side intent detection (lightweight, for immediate UI feedback) ─────
function detectIntentFromText(text) {
  const t = text.toLowerCase();
  // Check for order tracking first (high priority)
  if (/\border\b|\btrack|\bstatus\b|\bshipment|\bdeliver|\bord-\d/.test(t)) return 'ORDER_TRACKING';
  // Check for returns/policy
  if (/\breturn\b|\brefund|\bexchange|\bpolicy|\bshipping cost/.test(t)) return 'RETURNS_POLICY';
  // Check for product queries
  if (/\bprice|\bstock|\bspec|\bcompare|\bbuy|\bsell|\bproduct|\bitem/.test(t)) return 'PRODUCT_QUERY';
  // For very short messages (likely greetings or personal info), return null to let backend handle it
  if (text.split().length <= 3) {
    return null;
  }
  return null;
}

function detectIntentFromResponse(text) {
  return currentIntent; // keep current intent after response
}

// ── Simple Markdown renderer ──────────────────────────────────────────────────
function renderMarkdown(text) {
  let html = escapeHtml(text);

  // Bold **text** and __text__
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

  // Italic *text* and _text_
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

  // Inline code `code`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Tables (simple)
  html = renderTables(html);

  // Unordered lists (- item or * item)
  html = renderLists(html);

  // Paragraphs — split on double newlines
  html = html
    .split(/\n{2,}/)
    .map(para => {
      para = para.trim();
      if (!para) return '';
      // Don't wrap tables or lists in <p>
      if (para.startsWith('<table') || para.startsWith('<ul') || para.startsWith('<ol')) return para;
      return `<p>${para.replace(/\n/g, '<br />')}</p>`;
    })
    .join('');

  return html;
}

function renderTables(html) {
  // Match markdown table blocks: header row | separator row | data rows
  return html.replace(
    /((?:\|[^\n]+\|\n?)+)/g,
    (block) => {
      const lines = block.trim().split('\n').filter(l => l.trim());
      if (lines.length < 2) return block;
      // Check for separator line
      const sepIdx = lines.findIndex(l => /^\|[-:\s|]+\|$/.test(l.trim()));
      if (sepIdx < 0) return block;

      const headers = parseTableRow(lines[0]);
      const rows    = lines.slice(sepIdx + 1).map(parseTableRow);

      const thead = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>`;
      const tbody = `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>`;
      return `<table>${thead}${tbody}</table>`;
    }
  );
}

function parseTableRow(row) {
  return row.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
}

function renderLists(html) {
  // Process unordered lists (lines starting with - or * or •)
  return html.replace(/((?:^|\n)[•\-\*] .+)+/gm, (block) => {
    const items = block.trim().split('\n').map(l => {
      return `<li>${l.replace(/^[•\-\*] /, '').trim()}</li>`;
    });
    return `<ul>${items.join('')}</ul>`;
  });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer = null;

function showToast(message, type = '') {
  clearTimeout(toastTimer);
  $toast.textContent = message;
  $toast.className = `toast ${type} visible`;
  toastTimer = setTimeout(() => {
    $toast.className = 'toast';
  }, 4000);
}

// ── Event listeners ───────────────────────────────────────────────────────────

// Send on Enter (not Shift+Enter)
$messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const text = $messageInput.value.trim();
    if (text && !isStreaming) {
      sendMessage(text);
      $messageInput.value = '';
      $messageInput.style.height = '';
      updateCharCount('');
    }
  }
});

// Auto-resize textarea
$messageInput.addEventListener('input', () => {
  $messageInput.style.height = 'auto';
  $messageInput.style.height = Math.min($messageInput.scrollHeight, 160) + 'px';
  const val = $messageInput.value;
  updateCharCount(val);
  $sendBtn.disabled = !val.trim() || isStreaming;
});

function updateCharCount(val) {
  const len = val.length;
  $charCount.textContent = `${len} / 2000`;
  $charCount.className = 'char-count' + (len > 1800 ? ' danger' : len > 1500 ? ' warning' : '');
}

// Send button
$sendBtn.addEventListener('click', () => {
  const text = $messageInput.value.trim();
  if (text) {
    sendMessage(text);
    $messageInput.value = '';
    $messageInput.style.height = '';
    updateCharCount('');
  }
});

// New session button
$newSessionBtn.addEventListener('click', resetSession);

// Sidebar toggle
$sidebarToggle.addEventListener('click', () => {
  $sidebar.classList.toggle('collapsed');
  $sidebar.classList.toggle('open');
});

// Suggestion buttons
$suggestionBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const msg = btn.dataset.msg;
    if (!isStreaming && msg) {
      $messageInput.value = '';
      sendMessage(msg);
      // Close sidebar on mobile
      if (window.innerWidth < 640) {
        $sidebar.classList.remove('open');
      }
    }
  });
});

// ── Voice Client Logic (Assignment 3: Vosk ASR + Piper TTS) ─────────────────
let voiceWs = null;
let audioContext = null;
let micStream = null;
let scriptNode = null;
let isRecording = false;
let playbackContext = null;
let nextStartTime = 0;
let activeAudioSources = [];

async function toggleVoiceRecording() {
  if (isRecording) {
    finishVoiceSpeaking();
  } else {
    await startVoiceRecording();
  }
}

async function startVoiceRecording() {
  if (isStreaming) {
    showToast('Please wait for the current response to finish.', 'error');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Microphone access is not supported in this browser.', 'error');
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
  } catch (err) {
    showToast('Microphone access was denied or is unavailable.', 'error');
    return;
  }

  isRecording = true;
  if ($voiceBtn) $voiceBtn.classList.add('recording');
  if ($voiceBar) $voiceBar.style.display = 'flex';
  if ($voiceStateText) $voiceStateText.textContent = 'Listening…';
  if ($voiceTranscriptPreview) $voiceTranscriptPreview.textContent = 'Listening for speech…';
  if ($voiceStopBtn) $voiceStopBtn.style.display = 'none';

  const voiceWsUrl = `ws://${location.host}/ws/voice`;
  voiceWs = new WebSocket(voiceWsUrl);
  voiceWs.binaryType = 'arraybuffer';

  voiceWs.onopen = () => {
    voiceWs.send(JSON.stringify({ type: 'start', session_id: sessionId, tts: true }));
  };

  voiceWs.onmessage = (event) => {
    if (typeof event.data === 'string') {
      try {
        const frame = JSON.parse(event.data);
        handleVoiceFrame(frame);
      } catch (e) {
        console.error('Voice frame parse error:', e);
      }
    }
  };

  voiceWs.onerror = (e) => {
    console.error('Voice WS error:', e);
  };

  voiceWs.onclose = () => {
    cleanupRecordingNodes();
  };

  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(micStream);
    const bufferSize = 4096;
    scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

    scriptNode.onaudioprocess = (audioEvent) => {
      if (!isRecording || !voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;
      const inputData = audioEvent.inputBuffer.getChannelData(0);
      const inputRate = audioContext.sampleRate;
      const targetRate = 16000;

      let resampled;
      if (inputRate === targetRate) {
        resampled = inputData;
      } else {
        const ratio = inputRate / targetRate;
        const newLength = Math.round(inputData.length / ratio);
        resampled = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
          const pos = i * ratio;
          const index = Math.floor(pos);
          const frac = pos - index;
          const s0 = inputData[index] || 0;
          const s1 = inputData[index + 1] || s0;
          resampled[i] = s0 + frac * (s1 - s0);
        }
      }

      const pcm16 = new Int16Array(resampled.length);
      for (let i = 0; i < resampled.length; i++) {
        const s = Math.max(-1, Math.min(1, resampled[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      voiceWs.send(pcm16.buffer);
    };

    source.connect(scriptNode);
    scriptNode.connect(audioContext.destination);
  } catch (err) {
    console.error('AudioContext setup error:', err);
    cleanupRecordingNodes();
  }
}

function finishVoiceSpeaking() {
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'end_of_speech' }));
  }
  cleanupRecordingNodes();
  if ($voiceBar) $voiceStateText.textContent = 'Processing…';
}

function cancelVoiceRecording() {
  if (voiceWs) {
    try { voiceWs.close(); } catch (e) {}
  }
  cleanupRecordingNodes();
  if ($voiceBar) $voiceBar.style.display = 'none';
}

function cleanupRecordingNodes() {
  isRecording = false;
  if ($voiceBtn) $voiceBtn.classList.remove('recording');
  if (scriptNode) {
    try { scriptNode.disconnect(); } catch (e) {}
    scriptNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }
}

function handleVoiceFrame(frame) {
  const { type, content, text, data, name, ok, elapsed_ms } = frame;

  if (type === 'ready') {
    if ($voiceStateText) $voiceStateText.textContent = 'Listening…';
  } else if (type === 'partial_transcript') {
    if (text && $voiceTranscriptPreview) {
      $voiceTranscriptPreview.textContent = `"${text}…"`;
    }
  } else if (type === 'final_transcript') {
    if ($voiceBar) $voiceBar.style.display = 'none';
    cleanupRecordingNodes();
    appendUserMessage(text, /* isVoice */ true);
    scrollToBottom();
    // Start bot turn
    isStreaming = true;
    hideTypingIndicator();
    setStatus('typing');
    $currentBotBubble = appendBotMessage('', true);
    currentBotText = '';
  } else if (type === 'no_speech') {
    if ($voiceStateText) $voiceStateText.textContent = 'No speech detected';
    setTimeout(() => {
      cancelVoiceRecording();
    }, 1200);
  } else if (type === 'tool_call') {
    appendToolChip(name, frame.arguments);
  } else if (type === 'tool_result') {
    updateToolChip(name, ok, elapsed_ms);
  } else if (type === 'token') {
    currentBotText += content;
    renderStreamingContent($currentBotBubble, currentBotText);
    scrollToBottom();
  } else if (type === 'audio') {
    playPcm16Chunk(data, frame.sample_rate || 22050);
  } else if (type === 'audio_end') {
    // Spoken audio finished streaming
  } else if (type === 'metrics') {
    appendMetricsChip($currentBotBubble, frame);
  } else if (type === 'done') {
    finishStreaming(false);
  } else if (type === 'error') {
    showToast(content, 'error');
    finishStreaming(false);
  }
}

function playPcm16Chunk(base64Data, sampleRate) {
  if (!playbackContext) {
    playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
    nextStartTime = playbackContext.currentTime;
  }
  if (playbackContext.state === 'suspended') {
    playbackContext.resume();
  }

  if ($voiceStopBtn) $voiceStopBtn.style.display = 'inline-block';

  const binaryString = atob(base64Data);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }

  const audioBuffer = playbackContext.createBuffer(1, float32.length, sampleRate);
  audioBuffer.copyToChannel(float32, 0);

  const source = playbackContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(playbackContext.destination);

  const now = playbackContext.currentTime;
  const startTime = Math.max(now, nextStartTime);
  source.start(startTime);
  nextStartTime = startTime + audioBuffer.duration;

  activeAudioSources.push(source);
  source.onended = () => {
    const idx = activeAudioSources.indexOf(source);
    if (idx !== -1) activeAudioSources.splice(idx, 1);
    if (activeAudioSources.length === 0 && $voiceStopBtn) {
      $voiceStopBtn.style.display = 'none';
    }
  };
}

function stopAllAudioPlayback() {
  for (const s of activeAudioSources) {
    try { s.stop(); } catch (e) {}
  }
  activeAudioSources = [];
  if (playbackContext) {
    nextStartTime = playbackContext.currentTime;
  }
  if ($voiceStopBtn) $voiceStopBtn.style.display = 'none';
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'stop_audio' }));
  }
}

// ── Voice Event Listeners ────────────────────────────────────────────────────
if ($voiceBtn) $voiceBtn.addEventListener('click', toggleVoiceRecording);
if ($voiceDoneBtn) $voiceDoneBtn.addEventListener('click', finishVoiceSpeaking);
if ($voiceCancelBtn) $voiceCancelBtn.addEventListener('click', cancelVoiceRecording);
if ($voiceStopBtn) $voiceStopBtn.addEventListener('click', stopAllAudioPlayback);

// ── Initialise ────────────────────────────────────────────────────────────────
function init() {
  sessionId = loadOrCreateSessionId();
  $sessionInfo.textContent = `Session: ${sessionId.slice(0, 8)}…`;
  setStatus('connecting');
  connect();
  enableInput(false);
  loadHistoryList();
}

init();
