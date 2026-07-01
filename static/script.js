(() => {
  const chatArea = document.getElementById('chat-area');
  const messagesEl = document.getElementById('messages');
  const input = document.getElementById('text-input');
  const sendBtn = document.getElementById('send-btn');
  const resetBtn = document.getElementById('reset-btn');
  const timestamp = document.getElementById('timestamp');

  let nameSet = false;
  let loading = false;

  function scrollToBottom(smooth = true) {
    if (!smooth) {
      chatArea.scrollTop = chatArea.scrollHeight;
      return;
    }
    requestAnimationFrame(() => {
      chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    });
  }

  function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    if (role === 'bot') {
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/><path d="M9 14h6"/><path d="M12 11v6"/></svg>`;
      div.appendChild(avatar);
    }

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = text.replace(/\n/g, '<br>');
    div.appendChild(bubble);

    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function showTyping() {
    const div = document.createElement('div');
    div.className = 'message bot typing';
    div.id = 'typing-indicator';
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/><path d="M9 14h6"/><path d="M12 11v6"/></svg>`;
    div.appendChild(avatar);
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('span');
      dot.className = 'dot';
      bubble.appendChild(dot);
    }
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  }

  function updatePlaceholder() {
    input.placeholder = nameSet ? 'Type your message...' : 'Type your name...';
  }

  function updateTimestamp() {
    const now = new Date();
    const opts = { month: 'long', day: 'numeric', year: 'numeric' };
    timestamp.textContent = now.toLocaleDateString('en-US', opts);
  }

  let lastRequestId = 0;

  async function sendMessage(text) {
    if (!text.trim() || loading) return;
    loading = true;
    const reqId = ++lastRequestId;

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
      addMessage(data.reply, 'bot');
    } catch {
      removeTyping();
    }
  }

  function handleSend() {
    const text = input.value.trim();
    if (text) sendMessage(text);
  }

  function resetChat() {
    if (loading) return;
    fetch('/reset', { method: 'POST' }).catch(() => {});
    messagesEl.innerHTML = '';
    nameSet = false;
    lastRequestId++;
    updatePlaceholder();
    updateTimestamp();
    loadGreeting();
    input.focus();
  }

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  sendBtn.addEventListener('click', handleSend);

  resetBtn.addEventListener('click', (e) => {
    e.preventDefault();
    resetChat();
  });

  document.getElementById('close-btn').addEventListener('click', () => {
    window.close();
  });

  document.getElementById('emoji-btn').addEventListener('click', () => {
    input.focus();
  });

  document.getElementById('attach-btn').addEventListener('click', () => {
    input.focus();
  });

  let lastScrollHeight = 0;

  function onResize() {
    const height = chatArea.scrollHeight;
    if (height !== lastScrollHeight) {
      lastScrollHeight = height;
      scrollToBottom(false);
    }
  }

  const resizeObserver = new ResizeObserver(onResize);
  resizeObserver.observe(chatArea);

  let keyboardActive = false;

  if ('visualViewport' in window) {
    const origHeight = window.visualViewport.height;
    window.visualViewport.addEventListener('resize', () => {
      const diff = origHeight - window.visualViewport.height;
      keyboardActive = diff > 100;
      if (keyboardActive) {
        setTimeout(() => scrollToBottom(false), 100);
      }
    });
  }

  updateTimestamp();
  loadGreeting();
  input.focus();
})();
