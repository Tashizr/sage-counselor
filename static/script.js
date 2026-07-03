(() => {
  // ═══════════════════════════════════════════════
  // DOM Elements
  // ═══════════════════════════════════════════════
  const chatArea = document.getElementById('chat-area');
  const messagesEl = document.getElementById('messages');
  const input = document.getElementById('text-input');
  const sendBtn = document.getElementById('send-btn');
  const emptyState = document.getElementById('empty-state');
  const safetyBar = document.getElementById('safety-bar');
  const safetyDismiss = document.getElementById('safety-dismiss');
  const sidebar = document.getElementById('sidebar');
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebarClose = document.getElementById('sidebar-close');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
  const sidebarSearch = document.getElementById('sidebar-search');
  const newChatBtn = document.getElementById('new-chat-btn');
  const exportBtn = document.getElementById('export-btn');

  let nameSet = false;
  let loading = false;
  let lastRequestId = 0;

  // ═══════════════════════════════════════════════
  // Utilities
  // ═══════════════════════════════════════════════
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

  // ═══════════════════════════════════════════════
  // Safety Bar
  // ═══════════════════════════════════════════════
  safetyDismiss.addEventListener('click', () => {
    safetyBar.classList.add('hidden');
    document.querySelector('.app-layout').style.height = '100dvh';
  });

  // ═══════════════════════════════════════════════
  // Sidebar
  // ═══════════════════════════════════════════════
  function openSidebar() {
    sidebar.classList.add('open');
    sidebarOverlay.classList.add('active');
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('active');
  }

  sidebarToggle.addEventListener('click', openSidebar);
  sidebarClose.addEventListener('click', closeSidebar);
  sidebarOverlay.addEventListener('click', closeSidebar);

  // New chat
  newChatBtn.addEventListener('click', async () => {
    await fetch('/reset', { method: 'POST' });
    messagesEl.innerHTML = '';
    nameSet = false;
    emptyState.classList.remove('hidden');
    closeSidebar();
    loadGreeting();
  });

  // Search conversations
  sidebarSearch.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const items = document.querySelectorAll('.conv-item');
    items.forEach(item => {
      const text = item.querySelector('.conv-item-text').textContent.toLowerCase();
      item.style.display = text.includes(query) ? 'flex' : 'none';
    });
  });

  // ═══════════════════════════════════════════════
  // Messages
  // ═══════════════════════════════════════════════
  function addDateDivider() {
    const existing = messagesEl.querySelector('.date-divider');
    if (existing) return;
    const div = document.createElement('div');
    div.className = 'date-divider';
    div.textContent = 'Today';
    messagesEl.appendChild(div);
  }

  function getInitials(name) {
    if (!name) return '?';
    return name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
  }

  function addMessage(text, role) {
    emptyState.classList.add('hidden');

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const row = document.createElement('div');
    row.className = 'message-row';

    if (role === 'bot') {
      const avatar = document.createElement('div');
      avatar.className = 'avatar bot-avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>`;
      row.appendChild(avatar);
    } else {
      const avatar = document.createElement('div');
      avatar.className = 'avatar user-avatar';
      avatar.textContent = getInitials(nameSet ? 'User' : '');
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
    avatar.className = 'avatar bot-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>`;
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

  // ═══════════════════════════════════════════════
  // Textarea Auto-resize
  // ═══════════════════════════════════════════════
  function autoResize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  }

  input.addEventListener('input', autoResize);

  // ═══════════════════════════════════════════════
  // Send Message
  // ═══════════════════════════════════════════════
  async function sendMessage(text) {
    if (!text.trim() || loading) return;
    loading = true;
    const reqId = ++lastRequestId;

    addMessage(text, 'user');
    input.value = '';
    autoResize();
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
        input.placeholder = "Share what's on your mind...";
      }
    } catch {
      if (reqId === lastRequestId) {
        removeTyping();
        addMessage("Sorry, I couldn't reach the server. Please try again.", 'bot');
      }
    }

    loading = false;
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

  // ═══════════════════════════════════════════════
  // Suggestion Cards
  // ═══════════════════════════════════════════════
  document.querySelectorAll('.suggestion-card').forEach(card => {
    card.addEventListener('click', () => {
      const msg = card.getAttribute('data-message');
      if (msg) sendMessage(msg);
    });
  });

  // ═══════════════════════════════════════════════
  // Input Chips
  // ═══════════════════════════════════════════════
  document.querySelectorAll('.input-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const action = chip.getAttribute('data-action');
      if (action === 'breathe') openModal('breathing-modal');
      else if (action === 'mood') openModal('mood-modal');
      else if (action === 'crisis') openModal('crisis-modal');
      else if (action === 'ground') openModal('grounding-modal');
    });
  });

  // ═══════════════════════════════════════════════
  // Sidebar Tools
  // ═══════════════════════════════════════════════
  document.getElementById('tool-breathing')?.addEventListener('click', () => {
    closeSidebar();
    openModal('breathing-modal');
  });

  document.getElementById('tool-mood')?.addEventListener('click', () => {
    closeSidebar();
    openModal('mood-modal');
  });

  document.getElementById('tool-journal')?.addEventListener('click', () => {
    closeSidebar();
    sendMessage("I'd like to journal about how I'm feeling.");
  });

  document.getElementById('tool-crisis')?.addEventListener('click', () => {
    closeSidebar();
    openModal('crisis-modal');
  });

  // ═══════════════════════════════════════════════
  // Export
  // ═══════════════════════════════════════════════
  exportBtn.addEventListener('click', () => {
    const msgs = messagesEl.querySelectorAll('.message');
    let text = 'SAGE Conversation Export\n';
    text += '='.repeat(40) + '\n\n';
    msgs.forEach(msg => {
      const isUser = msg.classList.contains('user');
      const bubble = msg.querySelector('.bubble');
      const time = msg.querySelector('.message-time');
      const role = isUser ? 'You' : 'SAGE';
      text += `${role}`;
      if (time) text += ` (${time.textContent})`;
      text += `:\n${bubble?.textContent || ''}\n\n`;
    });
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `sage-conversation-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // ═══════════════════════════════════════════════
  // Modals
  // ═══════════════════════════════════════════════
  function openModal(id) {
    document.getElementById(id)?.classList.add('active');
  }

  function closeModal(id) {
    document.getElementById(id)?.classList.remove('active');
  }

  document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', () => {
      btn.closest('.modal-overlay')?.classList.remove('active');
    });
  });

  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.classList.remove('active');
    });
  });

  // ═══════════════════════════════════════════════
  // Breathing Exercise
  // ═══════════════════════════════════════════════
  let breathingInterval = null;

  document.getElementById('breathing-start')?.addEventListener('click', function () {
    const circle = document.getElementById('breathing-circle');
    const text = document.getElementById('breathing-text');
    const btn = this;
    const instruction = document.getElementById('breathing-instruction');

    if (breathingInterval) {
      clearInterval(breathingInterval);
      breathingInterval = null;
      circle.classList.remove('inhale', 'exhale');
      text.textContent = 'Breathe In';
      instruction.textContent = 'Follow the circle. Breathe in as it expands, out as it contracts.';
      btn.textContent = 'Start Exercise';
      return;
    }

    btn.textContent = 'Stop';
    let phase = 'inhale';
    let count = 4;

    function tick() {
      if (phase === 'inhale') {
        circle.classList.remove('exhale');
        circle.classList.add('inhale');
        text.textContent = `Breathe In`;
        instruction.textContent = `Hold... ${count}`;
        count--;
        if (count < 0) {
          phase = 'hold';
          count = 7;
        }
      } else if (phase === 'hold') {
        text.textContent = `Hold`;
        instruction.textContent = `Hold for ${count} seconds`;
        count--;
        if (count < 0) {
          phase = 'exhale';
          count = 8;
        }
      } else {
        circle.classList.remove('inhale');
        circle.classList.add('exhale');
        text.textContent = `Breathe Out`;
        instruction.textContent = `Breathe out slowly... ${count}`;
        count--;
        if (count < 0) {
          phase = 'inhale';
          count = 4;
        }
      }
    }

    tick();
    breathingInterval = setInterval(tick, 1000);
  });

  // ═══════════════════════════════════════════════
  // Mood Check-in
  // ═══════════════════════════════════════════════
  document.querySelectorAll('.mood-option').forEach(opt => {
    opt.addEventListener('click', () => {
      document.querySelectorAll('.mood-option').forEach(o => o.classList.remove('selected'));
      opt.classList.add('selected');
      const mood = opt.getAttribute('data-mood');
      setTimeout(() => {
        closeModal('mood-modal');
        const moodMessages = {
          great: "I'm feeling great today",
          good: "I'm feeling good",
          okay: "I'm feeling okay",
          low: "I'm feeling a bit low",
          bad: "I'm feeling bad",
          awful: "I'm feeling awful"
        };
        sendMessage(moodMessages[mood] || "I wanted to check in about my mood");
      }, 300);
    });
  });

  // ═══════════════════════════════════════════════
  // Grounding Exercise
  // ═══════════════════════════════════════════════
  const groundingSteps = [
    { n: 5, prompt: 'Name <strong>5 things</strong> you can <strong>see</strong> around you.' },
    { n: 4, prompt: 'Name <strong>4 things</strong> you can <strong>touch</strong> or feel.' },
    { n: 3, prompt: 'Name <strong>3 things</strong> you can <strong>hear</strong> right now.' },
    { n: 2, prompt: 'Name <strong>2 things</strong> you can <strong>smell</strong>.' },
    { n: 1, prompt: 'Name <strong>1 thing</strong> you can <strong>taste</strong>.' },
  ];

  let groundingStep = 0;

  document.getElementById('grounding-next')?.addEventListener('click', () => {
    groundingStep++;
    if (groundingStep >= groundingSteps.length) {
      closeModal('grounding-modal');
      groundingStep = 0;
      document.getElementById('grounding-progress-bar').style.width = '0%';
      document.getElementById('grounding-number').textContent = '5';
      document.getElementById('grounding-prompt').innerHTML = groundingSteps[0].prompt;
      sendMessage("I just completed the 5-4-3-2-1 grounding exercise.");
      return;
    }
    const step = groundingSteps[groundingStep];
    document.getElementById('grounding-number').textContent = step.n;
    document.getElementById('grounding-prompt').innerHTML = step.prompt;
    document.getElementById('grounding-progress-bar').style.width = `${((groundingStep + 1) / groundingSteps.length) * 100}%`;
  });

  // ═══════════════════════════════════════════════
  // Load Greeting
  // ═══════════════════════════════════════════════
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

  // ═══════════════════════════════════════════════
  // Keyboard Shortcuts
  // ═══════════════════════════════════════════════
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
      closeSidebar();
    }
  });

  // ═══════════════════════════════════════════════
  // Scroll Handling
  // ═══════════════════════════════════════════════
  if ('visualViewport' in window) {
    window.visualViewport.addEventListener('resize', () => {
      setTimeout(() => scrollToBottom(false), 150);
    });
  }

  const resizeObserver = new ResizeObserver(() => scrollToBottom(false));
  resizeObserver.observe(chatArea);

  // ═══════════════════════════════════════════════
  // Init
  // ═══════════════════════════════════════════════
  loadGreeting();
  input.focus();
})();
