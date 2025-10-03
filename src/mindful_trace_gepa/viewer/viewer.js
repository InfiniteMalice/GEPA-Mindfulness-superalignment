(function () {
  const data = window.__GEPA__ || {};
  const manifest = data.manifest || {};
  const settings = data.settings || {};
  const pageSize = Math.max(1, settings.pageSize || 200);
  const maxPoints = Math.max(1, settings.maxPoints || 5000);
  const manifestPath = settings.manifestPath || null;

  const tokens = data.tokens || [];
  const deception = data.deception || {};
  const paired = data.paired || {};
  const scoringData = data.scoring || {};

  const timelineList = document.getElementById("timeline-list");
  const controls = document.getElementById("timeline-controls");
  const infoBanner = document.getElementById("info-banner");
  const eventText = document.getElementById("event-text");
  const eventMeta = document.getElementById("event-meta");
  const tokenStrip = document.getElementById("token-strip");
  const tokenCanvas = document.getElementById("token-chart");
  const deceptionContainer = document.getElementById("deception-content");
  const scoringPanel = document.getElementById("scoring");
  const scoringSummary = document.getElementById("scoring-summary");
  const scoringTiers = document.getElementById("scoring-tiers");
  const scoringToggle = document.getElementById("scoring-toggle");
  const scoringRationales = document.getElementById("scoring-rationales");

  let pageEvents = data.trace || [];
  const baseEvents = data.trace || [];
  const fullTraceText = (baseEvents || [])
    .map((evt) => evt.content || evt.text || evt.final_answer || "")
    .join("\n");
  const pageCache = new Map();
  const shardCache = new Map();
  let currentPage = 0;
  let selectedIndex = 0;

  const shardOffsets = [];
  if (manifest && Array.isArray(manifest.shards)) {
    let offset = 0;
    manifest.shards.forEach((shard) => {
      const events = shard.events || 0;
      shardOffsets.push({
        path: shard.path,
        start: offset,
        end: offset + events,
      });
      offset += events;
    });
  }
  const totalEvents = manifest.total_events || baseEvents.length || (shardOffsets.length ? shardOffsets[shardOffsets.length - 1].end : 0);

  const infoMessages = [];
  function pushInfo(message) {
    if (!message || infoMessages.includes(message)) return;
    infoMessages.push(message);
    infoBanner.textContent = infoMessages.join(" ");
    infoBanner.style.display = infoMessages.length ? "block" : "none";
  }

  if (!tokens.length) {
    pushInfo("Token log not provided; charts will be sparse.");
  }
  if (manifestPath === null && manifest && manifest.shards && manifest.shards.length) {
    pushInfo("Manifest path missing; shard loading may fail depending on browser security settings.");
  }
  if (manifest && manifest.shards && manifest.shards.length) {
    pushInfo(`Loaded manifest with ${manifest.shards.length} shard(s).`);
  }

  function resolveShardPath(shardPath) {
    if (!manifestPath) return shardPath;
    const parts = manifestPath.split(/[\\/]/);
    parts.pop();
    const base = parts.join("/");
    if (!base) {
      return shardPath;
    }
    return `${base}/${shardPath}`;
  }

  async function loadShard(shard) {
    if (!shard || !shard.path) return [];
    if (shardCache.has(shard.path)) {
      return shardCache.get(shard.path);
    }
    if (shard.path.endsWith(".zst")) {
      pushInfo("Compressed shards (.zst) cannot be loaded in-browser. Decompress to JSONL for full viewing.");
      shardCache.set(shard.path, []);
      return [];
    }
    const target = resolveShardPath(shard.path);
    try {
      const response = await fetch(target);
      if (!response.ok) {
        pushInfo(`Failed to fetch shard ${shard.path}: ${response.status}`);
        shardCache.set(shard.path, []);
        return [];
      }
      const text = await response.text();
      const rows = text
        .split(/\n+/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch (err) {
            console.warn("Failed to parse shard line", err);
            return null;
          }
        })
        .filter(Boolean);
      shardCache.set(shard.path, rows);
      return rows;
    } catch (err) {
      pushInfo(`Unable to load shard ${shard.path}`);
      shardCache.set(shard.path, []);
      return [];
    }
  }

  async function loadPage(page) {
    if (pageCache.has(page)) {
      return pageCache.get(page);
    }
    const start = page * pageSize;
    const end = Math.min(start + pageSize, totalEvents);
    if (!manifest.shards || !manifest.shards.length) {
      const slice = baseEvents.slice(start, end);
      pageCache.set(page, slice);
      return slice;
    }
    const collected = [];
    for (const shard of shardOffsets) {
      if (shard.end <= start || shard.start >= end) {
        continue;
      }
      const shardRows = await loadShard(shard);
      if (!shardRows.length) {
        continue;
      }
      const from = Math.max(start, shard.start) - shard.start;
      const to = Math.min(end, shard.end) - shard.start;
      const slice = shardRows.slice(from, to);
      collected.push(...slice);
      if (collected.length >= end - start) {
        break;
      }
    }
    const finalSlice = collected.slice(0, Math.max(0, end - start));
    pageCache.set(page, finalSlice);
    return finalSlice;
  }

  let prevBtn;
  let nextBtn;
  let pageInfo;

  function initControls() {
    controls.innerHTML = "";
    prevBtn = document.createElement("button");
    prevBtn.textContent = "Prev";
    nextBtn = document.createElement("button");
    nextBtn.textContent = "Next";
    pageInfo = document.createElement("span");
    pageInfo.className = "page-info";
    controls.appendChild(prevBtn);
    controls.appendChild(pageInfo);
    controls.appendChild(nextBtn);
    prevBtn.addEventListener("click", () => gotoPage(currentPage - 1));
    nextBtn.addEventListener("click", () => gotoPage(currentPage + 1));
  }

  function updateControls() {
    const maxPage = Math.max(0, Math.ceil(totalEvents / pageSize) - 1);
    pageInfo.textContent = totalEvents
      ? `Page ${currentPage + 1} / ${maxPage + 1} (${totalEvents} events)`
      : "No events";
    prevBtn.disabled = currentPage <= 0;
    nextBtn.disabled = currentPage >= maxPage;
  }

  function renderTimeline() {
    timelineList.innerHTML = "";
    pageEvents.forEach((evt, index) => {
      const li = document.createElement("li");
      const label = evt.timestamp || evt.stage || evt.module || `Event ${index + 1}`;
      const globalIndex = currentPage * pageSize + index;
      li.textContent = `${globalIndex + 1}. ${label}`;
      li.addEventListener("click", () => selectEvent(index));
      if (index === selectedIndex) {
        li.classList.add("active");
      }
      timelineList.appendChild(li);
    });
    if (pageEvents.length === 0) {
      const empty = document.createElement("li");
      empty.textContent = "No events loaded for this page.";
      timelineList.appendChild(empty);
    }
  }

  function selectEvent(index) {
    if (!pageEvents[index]) {
      return;
    }
    selectedIndex = index;
    const evt = pageEvents[index];
    eventText.textContent = evt.content || evt.text || "";
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
    if (evt.flags) {
      eventMeta.innerHTML += `<div>Flags: ${JSON.stringify(evt.flags)}</div>`;
    }
    const items = timelineList.querySelectorAll("li");
    items.forEach((node) => node.classList.remove("active"));
    if (items[index]) {
      items[index].classList.add("active");
    }
  }

  function renderTokens() {
    tokenStrip.innerHTML = "";
    if (!tokens.length) {
      return;
    }
    const step = Math.max(1, Math.floor(tokens.length / maxPoints));
    const sampled = tokens.filter((_, idx) => idx % step === 0);
    sampled.forEach((token) => {
      const span = document.createElement("span");
      span.textContent = token.token;
      span.className = "token-chip" + (token.abstained ? " abstain" : "");
      span.title = `chunk ${token.chunk ?? 0} offset ${token.offset ?? 0} perplexity ${token.ppl ? token.ppl.toFixed(2) : "n/a"}`;
      tokenStrip.appendChild(span);
    });
    if (tokenCanvas && tokenCanvas.getContext) {
      const ctx = tokenCanvas.getContext("2d");
      ctx.clearRect(0, 0, tokenCanvas.width, tokenCanvas.height);
      if (!sampled.length) {
        return;
      }
      ctx.strokeStyle = "#1f3c88";
      ctx.beginPath();
      sampled.forEach((token, index) => {
        const x = (index / Math.max(sampled.length - 1, 1)) * tokenCanvas.width;
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
      pieces.push("<h3>Honest Chain</h3>");
      pieces.push(`<pre>${JSON.stringify(paired.honest_chain, null, 2)}</pre>`);
    }
    if (paired && paired.deceptive_chain) {
      pieces.push("<h3>Deceptive Chain</h3>");
      pieces.push(`<pre>${JSON.stringify(paired.deceptive_chain, null, 2)}</pre>`);
    }
    deceptionContainer.innerHTML = pieces.join("\n");
  }

  function spanSnippet(span) {
    if (!span) return "";
    if (!fullTraceText) return "";
    const start = Math.max(0, span.start || 0);
    const end = Math.min(fullTraceText.length, span.end || start + 1);
    const snippet = fullTraceText.slice(start, end).trim();
    if (!snippet) return "";
    return snippet.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderScoring() {
    if (!scoringPanel) return;
    if (!scoringData || Object.keys(scoringData).length === 0) {
      scoringPanel.style.display = "none";
      return;
    }
    scoringPanel.style.display = "block";
    const final = scoringData.final || {};
    const confidence = scoringData.confidence || {};
    const reasons = scoringData.reasons || [];
    const escalate = !!scoringData.escalate;

    scoringSummary.innerHTML = "";
    Object.entries(final).forEach(([dim, score]) => {
      const span = document.createElement("span");
      span.className = "score-chip" + (escalate ? " escalate" : "");
      const conf = confidence[dim] !== undefined ? ` <span class="confidence">(${(confidence[dim] * 100).toFixed(0)}%)</span>` : "";
      span.innerHTML = `${dim}: ${score}${conf}`;
      scoringSummary.appendChild(span);
    });
    if (reasons.length) {
      const list = document.createElement("ul");
      list.style.margin = "0.5rem 0 0";
      reasons.forEach((reason) => {
        const li = document.createElement("li");
        li.textContent = reason;
        list.appendChild(li);
      });
      scoringSummary.appendChild(list);
    }

    const tiers = scoringData.per_tier || [];
    if (tiers.length) {
      const table = document.createElement("table");
      const header = document.createElement("tr");
      header.innerHTML = `<th>Tier</th>${Object.keys(final)
        .map((dim) => `<th>${dim}</th>`)
        .join("")}`;
      table.appendChild(header);
      tiers.forEach((tier) => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${tier.tier}</td>${Object.keys(final)
          .map((dim) => {
            const score = tier.scores ? tier.scores[dim] : "-";
            const conf = tier.confidence ? tier.confidence[dim] : 0;
            return `<td>${score} <span class="confidence">${(conf * 100).toFixed(0)}%</span></td>`;
          })
          .join("")}`;
        table.appendChild(row);
      });
      scoringTiers.innerHTML = "";
      scoringTiers.appendChild(table);
    }

    const judgeTier = tiers.find((tier) => tier.tier === "judge");
    if (!judgeTier || !scoringToggle || !scoringRationales) {
      if (scoringToggle) {
        scoringToggle.style.display = "none";
      }
      return;
    }
    const rationals = (judgeTier.meta && judgeTier.meta.rationales) || {};
    const spans = (judgeTier.meta && judgeTier.meta.spans) || {};
    const dl = document.createElement("dl");
    Object.entries(rationals).forEach(([dim, rationale]) => {
      const dt = document.createElement("dt");
      dt.textContent = `${dim}: ${rationale}`;
      const dd = document.createElement("dd");
      const spanList = spans[dim] || [];
      spanList.forEach((span) => {
        const snippet = spanSnippet(span);
        if (!snippet) {
          return;
        }
        const mark = document.createElement("span");
        mark.className = "span-snippet";
        mark.textContent = snippet;
        dd.appendChild(mark);
      });
      dl.appendChild(dt);
      if (dd.childNodes.length) {
        dl.appendChild(dd);
      }
    });
    scoringRationales.innerHTML = "";
    scoringRationales.appendChild(dl);

    let shown = false;
    scoringToggle.addEventListener("click", () => {
      shown = !shown;
      if (shown) {
        scoringRationales.classList.remove("hidden");
        scoringToggle.textContent = "Hide rationales";
      } else {
        scoringRationales.classList.add("hidden");
        scoringToggle.textContent = "Show rationales";
      }
    });
  }

  async function gotoPage(page) {
    const maxPage = Math.max(0, Math.ceil(Math.max(totalEvents, baseEvents.length) / pageSize) - 1);
    currentPage = Math.min(Math.max(page, 0), maxPage);
    pageEvents = await loadPage(currentPage);
    selectedIndex = 0;
    renderTimeline();
    selectEvent(0);
    updateControls();
  }

  async function init() {
    initControls();
    renderTokens();
    renderDeception();
    renderScoring();
    if (totalEvents === 0 && !baseEvents.length) {
      pushInfo("No trace events available. Did the run complete?");
    }
    await gotoPage(0);
  }

  init();
})();
