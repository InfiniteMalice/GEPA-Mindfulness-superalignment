(function () {
  const data = window.__GEPA__ || {};
  const events = data.trace || [];
  const tokens = data.tokens || [];
  const deception = data.deception || {};
  const paired = data.paired || {};

  const timelineList = document.getElementById("timeline-list");
  const eventText = document.getElementById("event-text");
  const eventMeta = document.getElementById("event-meta");
  const tokenStrip = document.getElementById("token-strip");
  const tokenCanvas = document.getElementById("token-chart");
  const deceptionContainer = document.getElementById("deception-content");

  function renderTimeline() {
    timelineList.innerHTML = "";
    events.forEach((evt, index) => {
      const li = document.createElement("li");
      li.textContent = `${evt.timestamp || index} – ${evt.stage || evt.module}`;
      li.addEventListener("click", () => selectEvent(index));
      if (index === 0) {
        li.classList.add("active");
      }
      timelineList.appendChild(li);
    });
    if (events.length > 0) {
      selectEvent(0);
    }
  }

  function selectEvent(index) {
    const evt = events[index];
    eventText.textContent = evt.content || "";
    eventMeta.innerHTML = "";
    const badges = (evt.gepa_hits || []).map((hit) => `<span class="badge">${hit}</span>`).join("");
    if (badges) {
      eventMeta.innerHTML += `<div>GEPA badges: ${badges}</div>`;
    }
    if (evt.principle_scores) {
      eventMeta.innerHTML += `<div>Principles: ${JSON.stringify(evt.principle_scores)}</div>`;
    }
    if (evt.imperative_scores) {
      eventMeta.innerHTML += `<div>Imperatives: ${JSON.stringify(evt.imperative_scores)}</div>`;
    }
    if (evt.context) {
      eventMeta.innerHTML += `<div>Context: ${evt.context}</div>`;
    }
    const items = timelineList.querySelectorAll("li");
    items.forEach((node) => node.classList.remove("active"));
    if (items[index]) {
      items[index].classList.add("active");
    }
  }

  function renderTokens() {
    tokenStrip.innerHTML = "";
    tokens.forEach((token) => {
      const span = document.createElement("span");
      span.textContent = token.token;
      span.className = "token-chip" + (token.abstained ? " abstain" : "");
      tokenStrip.appendChild(span);
    });
    if (tokenCanvas && tokenCanvas.getContext) {
      const ctx = tokenCanvas.getContext("2d");
      ctx.clearRect(0, 0, tokenCanvas.width, tokenCanvas.height);
      ctx.strokeStyle = "#1f3c88";
      ctx.beginPath();
      tokens.forEach((token, index) => {
        const x = (index / Math.max(tokens.length - 1, 1)) * tokenCanvas.width;
        const y = tokenCanvas.height - (token.conf || 0) * tokenCanvas.height * 0.8;
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
  }

  function renderDeception() {
    if (!deceptionContainer) return;
    const pieces = [];
    if (deception && Object.keys(deception).length > 0) {
      pieces.push(`<div><strong>Deception Score:</strong> ${(deception.score ?? 0).toFixed(2)}</div>`);
      if (deception.reasons) {
        pieces.push(`<ul>${deception.reasons.map((r) => `<li>${r}</li>`).join("")}</ul>`);
      }
    }
    if (paired && paired.honest_chain) {
      pieces.push('<h3>Honest Chain</h3>');
      pieces.push(`<pre>${JSON.stringify(paired.honest_chain, null, 2)}</pre>`);
    }
    if (paired && paired.deceptive_chain) {
      pieces.push('<h3>Deceptive Chain</h3>');
      pieces.push(`<pre>${JSON.stringify(paired.deceptive_chain, null, 2)}</pre>`);
    }
    deceptionContainer.innerHTML = pieces.join("\n");
  }

  renderTimeline();
  renderTokens();
  renderDeception();
})();
