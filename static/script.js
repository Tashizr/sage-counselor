(() => {
  const chatArea = document.getElementById('chat-area');
  const messagesEl = document.getElementById('messages');
  const input = document.getElementById('text-input');
  const sendBtn = document.getElementById('send-btn');
  const chipsEl = document.getElementById('chips');

  let nameSet = false;
  let loading = false;
  let lastRequestId = 0;

  const SUGGESTED_CHIPS = [
    "I'm feeling anxious",
    "I need someone to talk to",
    "I'm having dark thoughts",
    "Help me calm down",
  ];

  function pad(n) { return String(n).padStart(2, '0'); }

  function now() {
    const d = new Date();
    const h = d.getHours();
    const m = d.getMinutes();
    const ampm = h >= 12 ? 'PM' : 'AM';
    return `${h % 12 || 12}:${pad(m)} ${ampm}`;
  }

  function scrollToBottom(smooth = true) {
    requestAnimationFrame(() => {
      if (smooth) {
        chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
      } else {
        chatArea.scrollTop = chatArea.scrollHeight;
      }
    });
  }

  function addDateDivider() {
    const existing = messagesEl.querySelector('.date-divider');
    if (existing) return;
    const div = document.createElement('div');
    div.className = 'date-divider';
    div.textContent = 'Today';
    messagesEl.appendChild(div);
  }

  function addMessage(text, role) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const row = document.createElement('div');
    row.className = 'message-row';

    if (role === 'bot') {
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/><path d="M9 14h6"/><path d="M12 11v6"/></svg>`;
      row.appendChild(avatar);
    }

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    row.appendChild(bubble);

    msgDiv.appendChild(row);

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = now();
    msgDiv.appendChild(time);

    messagesEl.appendChild(msgDiv);
    scrollToBottom();
  }

  function showTyping() {
    const div = document.createElement('div');
    div.className = 'message bot typing';
    div.id = 'typing-indicator';

    const row = document.createElement('div');
    row.className = 'message-row';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/><path d="M9 14h6"/><path d="M12 11v6"/></svg>`;
    row.appendChild(avatar);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('span');
      dot.className = 'dot';
      bubble.appendChild(dot);
    }
    row.appendChild(bubble);
    div.appendChild(row);
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  function updatePlaceholder() {
    input.placeholder = nameSet ? "Share what's on your mind..." : 'What should I call you?';
  }

  function setChips(chips) {
    chipsEl.innerHTML = '';
    chips.forEach(text => {
      const btn = document.createElement('button');
      btn.className = 'chip';
      btn.textContent = text;
      btn.type = 'button';
      btn.addEventListener('click', () => sendMessage(text));
      chipsEl.appendChild(btn);
    });
  }

  async function sendMessage(text) {
    if (!text.trim() || loading) return;
    loading = true;
    const reqId = ++lastRequestId;

    chipsEl.innerHTML = '';
    addMessage(text, 'user');
    input.value = '';
    showTyping();

    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await resp.json();

      if (reqId !== lastRequestId) return;

      removeTyping();
      addMessage(data.reply, 'bot');

      if (data.name_set && !nameSet) {
        nameSet = true;
        updatePlaceholder();
      }
      if (nameSet) {
        setChips(SUGGESTED_CHIPS);
      }
    } catch {
      if (reqId === lastRequestId) {
        removeTyping();
        addMessage("Sorry, I couldn't reach the server. Please try again.", 'bot');
      }
    }

    loading = false;
  }

  async function loadGreeting() {
    showTyping();
    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: '' }),
      });
      const data = await resp.json();
      removeTyping();
      addDateDivider();
      addMessage(data.reply, 'bot');
    } catch {
      removeTyping();
    }
  }

  function handleSend() {
    const text = input.value.trim();
    if (text) sendMessage(text);
  }

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  sendBtn.addEventListener('click', handleSend);

  document.getElementById('emoji-btn').addEventListener('click', () => {
    input.focus();
  });

  document.getElementById('crisis-btn').addEventListener('click', (e) => {
    e.stopPropagation();
  });

  if ('visualViewport' in window) {
    window.visualViewport.addEventListener('resize', () => {
      setTimeout(() => scrollToBottom(false), 150);
    });
  }

  const resizeObserver = new ResizeObserver(() => scrollToBottom(false));
  resizeObserver.observe(chatArea);

  loadGreeting();
  input.focus();
})();
