(function () {
  function loadData() {
    const node = document.getElementById("gepa-data");
    if (!node) return {};
    try {
      return JSON.parse(node.textContent || "{}");
    } catch (err) {
      console.warn("Failed to parse embedded GEPA data", err);
      return {};
    }
  }

  const data = loadData();
  const manifest = data.manifest || {};
  const settings = data.settings || {};
  const pageSize = Math.max(1, settings.pageSize || 200);
  const maxPoints = Math.max(1, settings.maxPoints || 5000);
  const manifestPath = settings.manifestPath || null;

  const tokens = data.tokens || [];
  const deception = data.deception || {};
  const dualPath = data.dualPath || data.paired || {};
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

  function clear(node) {
    if (node) node.replaceChildren();
  }

  function appendText(parent, tagName, text, className) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    node.textContent = text;
    parent.appendChild(node);
    return node;
  }

  function appendLabeledValue(parent, label, value) {
    const row = document.createElement("div");
    appendText(row, "strong", `${label}: `);
    row.appendChild(document.createTextNode(String(value)));
    parent.appendChild(row);
    return row;
  }

  function appendJsonPre(parent, value) {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(value, null, 2);
    parent.appendChild(pre);
    return pre;
  }

  let pageEvents = data.trace || [];
  const baseEvents = data.trace || [];
  let eventTypeFilter = "all";
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
    clear(controls);
    const typeSelect = document.createElement("select");
    const eventTypes = ["all", ...Array.from(new Set(baseEvents.map((evt) => evt.event_type || evt.stage || "legacy_trace_event"))).sort()];
    eventTypes.forEach((type) => {
      const option = document.createElement("option");
      option.value = type;
      option.textContent = type;
      typeSelect.appendChild(option);
    });
    typeSelect.addEventListener("change", () => {
      eventTypeFilter = typeSelect.value;
      gotoPage(0);
    });
    prevBtn = document.createElement("button");
    prevBtn.textContent = "Prev";
    nextBtn = document.createElement("button");
    nextBtn.textContent = "Next";
    pageInfo = document.createElement("span");
    pageInfo.className = "page-info";
    controls.appendChild(typeSelect);
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

  function visiblePageEvents() {
    return pageEvents.filter((evt) => {
      const type = evt.event_type || evt.stage || "legacy_trace_event";
      return eventTypeFilter === "all" || type === eventTypeFilter;
    });
  }

  function renderTimeline() {
    clear(timelineList);
    const visibleEvents = visiblePageEvents();
    visibleEvents.forEach((evt, index) => {
      const li = document.createElement("li");
      const type = evt.event_type || evt.stage || evt.module || "event";
      const label = `${type} ${evt.timestamp || evt.stage || evt.module || `Event ${index + 1}`}`;
      const globalIndex = currentPage * pageSize + index;
      li.textContent = `${globalIndex + 1}. ${label}`;
      li.addEventListener("click", () => selectEventByObject(evt, index));
      if (index === selectedIndex) {
        li.classList.add("active");
      }
      timelineList.appendChild(li);
    });
    if (visibleEvents.length === 0) {
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
    renderSelectedEvent(evt, index);
  }

  function selectEventByObject(evt, index) {
    selectedIndex = index;
    renderSelectedEvent(evt, index);
  }

  function renderSelectedEvent(evt, index) {
    eventText.textContent = evt.content || evt.text || "";
    clear(eventMeta);
    if (evt.event_type) {
      appendLabeledValue(eventMeta, "Event type", evt.event_type);
    }
    if (evt.event_id) {
      appendLabeledValue(eventMeta, "Event ID", evt.event_id);
    }
    if (evt.payload && Object.keys(evt.payload).length) {
      appendJsonPre(eventMeta, evt.payload);
    }
    const badges = evt.gepa_hits || [];
    if (badges.length) {
      const row = document.createElement("div");
      appendText(row, "strong", "GEPA badges: ");
      badges.forEach((hit) => appendText(row, "span", hit, "badge"));
      eventMeta.appendChild(row);
    }
    if (evt.telemetry_mode || evt.telemetry_available !== undefined) {
      const mode = evt.telemetry_mode || "unknown";
      const available = evt.telemetry_available === true ? "available" : "unavailable/synthetic";
      appendLabeledValue(eventMeta, "Telemetry", `${mode} (${available})`);
    }
    if (evt.principle_scores) {
      appendLabeledValue(eventMeta, "Principles", JSON.stringify(evt.principle_scores));
    }
    if (evt.imperative_scores) {
      appendLabeledValue(eventMeta, "Imperatives", JSON.stringify(evt.imperative_scores));
    }
    if (evt.context) {
      appendLabeledValue(eventMeta, "Context", evt.context);
    }
    if (evt.flags) {
      appendLabeledValue(eventMeta, "Flags", JSON.stringify(evt.flags));
    }
    const items = timelineList.querySelectorAll("li");
    items.forEach((node) => node.classList.remove("active"));
    if (items[index]) {
      items[index].classList.add("active");
    }
  }

  function selectFirstVisibleEvent() {
    const visibleEvents = visiblePageEvents();
    if (!visibleEvents.length) {
      eventText.textContent = "";
      clear(eventMeta);
      return;
    }
    selectEventByObject(visibleEvents[0], 0);
  }

  function renderTokens() {
    clear(tokenStrip);
    if (!tokens.length) {
      return;
    }
    const step = Math.max(1, Math.floor(tokens.length / maxPoints));
    const sampled = tokens.filter((_, idx) => idx % step === 0);
    sampled.forEach((token) => {
      const span = document.createElement("span");
      span.textContent = token.token;
      span.className = "token-chip" + (token.abstained ? " abstain" : "");
      const mode = token.telemetry_mode || "legacy";
      const measured = token.telemetry_available === true ? "measured" : "synthetic/unavailable";
      span.title = `chunk ${token.chunk ?? 0} offset ${token.offset ?? 0} perplexity ${token.ppl ? token.ppl.toFixed(2) : "n/a"} telemetry ${mode} ${measured}`;
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
    clear(deceptionContainer);
    if (deception && Object.keys(deception).length > 0) {
      appendLabeledValue(deceptionContainer, "Deception Score", (deception.score ?? 0).toFixed(2));
      if (deception.reasons) {
        const list = document.createElement("ul");
        deception.reasons.forEach((reason) => appendText(list, "li", reason));
        deceptionContainer.appendChild(list);
      }
    }
    if (dualPath && dualPath.honest_chain) {
      appendText(deceptionContainer, "h3", "Honest Chain");
      appendJsonPre(deceptionContainer, dualPath.honest_chain);
    }
    if (dualPath && dualPath.deceptive_chain) {
      appendText(deceptionContainer, "h3", "Deceptive Chain");
      appendJsonPre(deceptionContainer, dualPath.deceptive_chain);
    }
    if (deception && deception.probe) {
      appendText(deceptionContainer, "h3", "Deception Probe");
      appendJsonPre(deceptionContainer, deception.probe);
    }
    if (deception && deception.summary) {
      appendText(deceptionContainer, "h3", "Deception Summary");
      appendJsonPre(deceptionContainer, deception.summary);
    }
  }

  function spanSnippet(span) {
    if (!span) return "";
    if (!fullTraceText) return "";
    const start = Math.max(0, span.start || 0);
    const end = Math.min(fullTraceText.length, span.end || start + 1);
    const snippet = fullTraceText.slice(start, end).trim();
    if (!snippet) return "";
    return snippet;
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

    clear(scoringSummary);
    Object.entries(final).forEach(([dim, score]) => {
      const span = document.createElement("span");
      span.className = "score-chip" + (escalate ? " escalate" : "");
      span.appendChild(document.createTextNode(`${dim}: ${score}`));
      if (confidence[dim] !== undefined) {
        span.appendChild(document.createTextNode(" "));
        appendText(span, "span", `(${(confidence[dim] * 100).toFixed(0)}%)`, "confidence");
      }
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
      appendText(header, "th", "Tier");
      Object.keys(final).forEach((dim) => appendText(header, "th", dim));
      table.appendChild(header);
      tiers.forEach((tier) => {
        const row = document.createElement("tr");
        appendText(row, "td", tier.tier);
        Object.keys(final).forEach((dim) => {
          const cell = document.createElement("td");
          const score = tier.scores ? tier.scores[dim] : "-";
          const conf = tier.confidence ? tier.confidence[dim] : 0;
          cell.appendChild(document.createTextNode(`${score} `));
          appendText(cell, "span", `${(conf * 100).toFixed(0)}%`, "confidence");
          row.appendChild(cell);
        });
        table.appendChild(row);
      });
      clear(scoringTiers);
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
    clear(scoringRationales);
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
    selectFirstVisibleEvent();
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
