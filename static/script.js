/* ── Mongolian AI Chatbot — script.js ───────────────────────────────────────
   Handles: sending messages, rendering bubbles, typing indicator,
            textarea auto-resize, mobile sidebar, clear chat.
──────────────────────────────────────────────────────────────────────────── */

const chatArea  = document.getElementById("chatArea");
const userInput = document.getElementById("userInput");
const sendBtn   = document.getElementById("sendBtn");
const clearBtn  = document.getElementById("clearBtn");
const clearMob  = document.getElementById("clearMobile");
const menuToggle= document.getElementById("menuToggle");
const sidebar   = document.querySelector(".sidebar");

// ── Auto-resize textarea ──────────────────────────────────────────────────
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 150) + "px";
});

// ── Send on Enter (Shift+Enter = new line) ────────────────────────────────
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener("click", sendMessage);

// ── Mobile sidebar toggle ─────────────────────────────────────────────────
menuToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
document.addEventListener("click", (e) => {
  if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
    sidebar.classList.remove("open");
  }
});

// ── Clear chat ────────────────────────────────────────────────────────────
async function clearChat() {
  if (!confirm("Бүх яриаг устгах уу?")) return;
  await fetch("/clear", { method: "POST" });
  chatArea.innerHTML = "";
  // Re-show welcome screen
  chatArea.innerHTML = `
    <div class="welcome-screen" id="welcomeScreen">
      <div class="welcome-icon">🤖</div>
      <h1 class="welcome-title">Монгол AI Туслагч</h1>
      <p class="welcome-sub">Монгол, Англи, эсвэл Latin үсгээр асуул</p>
      <div class="suggestion-chips">
        <button class="chip" onclick="sendChip('Сайн уу!')">Сайн уу!</button>
        <button class="chip" onclick="sendChip('Python гэж юу вэ?')">Python гэж юу вэ?</button>
        <button class="chip" onclick="sendChip('Амжилтын нууц юу вэ?')">Амжилтын нууц юу вэ?</button>
        <button class="chip" onclick="sendChip('Монгол улсын тухай хэлж өг')">Монгол улс</button>
      </div>
    </div>`;
}
clearBtn.addEventListener("click", clearChat);
clearMob.addEventListener("click", clearChat);

// ── Send chip suggestion ──────────────────────────────────────────────────
function sendChip(text) {
  userInput.value = text;
  sendMessage();
}

// ── Core send function ────────────────────────────────────────────────────
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Hide welcome screen if still visible
  const welcome = document.getElementById("welcomeScreen");
  if (welcome) welcome.remove();

  // Render user bubble immediately
  const pair = createMessagePair();
  appendUserBubble(pair, text);

  // Reset input
  userInput.value = "";
  userInput.style.height = "auto";
  setBusy(true);

  // Show typing indicator
  const typingRow = appendTypingIndicator(pair);

  // Scroll to bottom
  scrollBottom();

  try {
    const res = await fetch("/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text }),
    });

    if (!res.ok) throw new Error("Серверийн алдаа");

    const data = await res.json();

    // Remove typing indicator, show real answer
    typingRow.remove();
    appendAIBubble(pair, data.response);
  } catch (err) {
    typingRow.remove();
    appendAIBubble(pair, "⚠️ Алдаа гарлаа. Серверт холбогдож чадсангүй.");
    console.error(err);
  } finally {
    setBusy(false);
    scrollBottom();
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────

/** Create and append a fresh message-pair container. */
function createMessagePair() {
  const div = document.createElement("div");
  div.className = "message-pair";
  chatArea.appendChild(div);
  return div;
}

/** Append the user's bubble into a pair. */
function appendUserBubble(pair, text) {
  const msg = document.createElement("div");
  msg.className = "message user-msg";
  msg.innerHTML = `
    <div class="bubble user-bubble">${escapeHtml(text)}</div>
    <span class="msg-time">${now()}</span>`;
  pair.appendChild(msg);
}

/** Append animated typing indicator into a pair; returns the element. */
function appendTypingIndicator(pair) {
  const msg = document.createElement("div");
  msg.className = "message ai-msg";
  msg.innerHTML = `
    <div class="ai-avatar">🤖</div>
    <div class="typing-bubble">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      <span style="margin-left:6px;font-size:12px;">AI бичиж байна...</span>
    </div>`;
  pair.appendChild(msg);
  return msg;
}

/** Append the AI's answer bubble. */
function appendAIBubble(pair, text) {
  const msg = document.createElement("div");
  msg.className = "message ai-msg";
  msg.innerHTML = `
    <div class="ai-avatar">🤖</div>
    <div class="bubble ai-bubble">${formatText(text)}</div>`;
  pair.appendChild(msg);
}

/** Disable/enable send button while waiting. */
function setBusy(busy) {
  sendBtn.disabled = busy;
  userInput.disabled = busy;
}

/** Scroll chat area to bottom. */
function scrollBottom() {
  chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: "smooth" });
}

/** Simple HTML escape to prevent XSS. */
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Format AI text: escape HTML, convert emoji numbers + newlines to nice HTML.
 * Also converts numbered list patterns (1️⃣ … 5️⃣).
 */
function formatText(str) {
  return escapeHtml(str)
    .replace(/\n/g, "<br>");
}

/** Return current time as HH:MM string. */
function now() {
  return new Date().toLocaleTimeString("mn-MN", { hour: "2-digit", minute: "2-digit" });
}

// ── Auto-scroll on page load ──────────────────────────────────────────────
window.addEventListener("load", scrollBottom);
