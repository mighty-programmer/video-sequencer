const state = {
  settings: {},
  server: {},
  benchmarks: [],
  cache: null,
  jobs: [],
  sessions: [],
  currentSession: null,
  activeSegmentId: null,
  selectedJobId: null,
  movementPreference: "auto",
};

const $ = (id) => document.getElementById(id);

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = "Request failed.";
    const clone = response.clone();
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch {
      detail = await clone.text();
    }
    throw new Error(detail);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response;
}

function mediaUrl(path) {
  return path ? `/api/media?path=${encodeURIComponent(path)}` : "";
}

function setStatus(id, message, isError = false) {
  const target = $(id);
  if (!target) return;
  target.textContent = message || "";
  target.style.color = isError ? "#b44343" : "";
}

function csvList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function asNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function shortText(text, max = 120) {
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function benchmarkOptionsHtml(includeCustom = false) {
  const options = state.benchmarks
    .map((benchmark) => `<option value="${benchmark.number}">Benchmark ${benchmark.number} · ${benchmark.title || benchmark.name}</option>`)
    .join("");
  return includeCustom
    ? `<option value="">Custom Paths</option>${options}`
    : options;
}

function syncBenchmarkSelects() {
  [
    "sessionBenchmark",
    "quickBenchmark",
    "fullBenchmark",
    "openclipGridBenchmark",
    "videoprismGridBenchmark",
    "wavGridBenchmark",
  ].forEach((id) => {
    const select = $(id);
    if (!select) return;
    const current = select.value;
    select.innerHTML = benchmarkOptionsHtml(id === "sessionBenchmark" || id === "fullBenchmark");
    if ([...select.options].some((option) => option.value === current)) {
      select.value = current;
    }
  });
}

function renderBenchmarks() {
  syncBenchmarkSelects();
  const container = $("benchmarkList");
  if (!container) return;
  container.innerHTML = "";

  state.benchmarks.forEach((benchmark) => {
    const node = document.createElement("div");
    node.className = "benchmark-item";
    node.innerHTML = `
      <div>
        <strong>Benchmark ${benchmark.number}</strong>
        <div class="muted">${benchmark.title || benchmark.name || ""}</div>
        <div class="muted">${benchmark.video_count} clips · ${benchmark.segment_count} segments</div>
      </div>
      <div class="launch-actions">
        <button class="ghost" data-use-benchmark="${benchmark.number}">Use In Editor</button>
        <a class="ghost" href="/api/benchmarks/${benchmark.number}/download">Download</a>
        <button class="ghost" data-delete-benchmark="${benchmark.number}">Delete</button>
      </div>
    `;
    container.appendChild(node);
  });

  container.querySelectorAll("[data-use-benchmark]").forEach((button) => {
    button.addEventListener("click", () => {
      $("sessionBenchmark").value = button.dataset.useBenchmark;
      switchTab("editor");
    });
  });

  container.querySelectorAll("[data-delete-benchmark]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!confirm(`Delete benchmark ${button.dataset.deleteBenchmark}?`)) return;
      try {
        await api(`/api/benchmarks/${button.dataset.deleteBenchmark}`, { method: "DELETE" });
        await refreshBootstrap();
      } catch (error) {
        setStatus("benchmarkUploadStatus", error.message, true);
      }
    });
  });
}

function renderCache() {
  const list = $("cacheList");
  if (!list || !state.cache) return;
  list.innerHTML = `
    <div class="cache-item">
      <div>
        <strong>${state.cache.cache_dir}</strong>
        <div class="muted">${state.cache.total_size_mb} MB total</div>
      </div>
    </div>
  `;

  state.cache.entries.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "cache-item";
    item.innerHTML = `
      <div>
        <strong>${entry.name}</strong>
        <div class="muted">${entry.type} · ${entry.size_mb} MB</div>
      </div>
    `;
    list.appendChild(item);
  });
}

function renderJobs() {
  const list = $("jobList");
  const log = $("jobLog");
  if (!list || !log) return;

  list.innerHTML = "";
  state.jobs.forEach((job) => {
    const node = document.createElement("div");
    node.className = `job-item ${job.status}`;
    node.innerHTML = `
      <div>
        <strong>${job.name}</strong>
        <div class="muted">${job.status} · ${job.created_at}</div>
      </div>
      <div class="launch-actions">
        <button class="ghost" data-job-id="${job.job_id}">View Log</button>
      </div>
    `;
    list.appendChild(node);
  });

  list.querySelectorAll("[data-job-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedJobId = button.dataset.jobId;
      const job = await api(`/api/jobs/${state.selectedJobId}`);
      log.textContent = job.log || "No log output yet.";
    });
  });

  if (state.selectedJobId) {
    const selected = state.jobs.find((job) => job.job_id === state.selectedJobId);
    if (selected) {
      log.textContent = selected.log || "No log output yet.";
    }
  }
}

function renderSettings() {
  $("settingGpu").value = state.settings.gpu_device || "";
  $("settingOutput").value = state.settings.output || "";
  $("settingCache").value = state.settings.cache_dir || "";
  $("settingBenchmarksDir").value = state.settings.benchmarks_dir || "";
  $("settingHostname").value = state.settings.server_hostname || "";
}

function renderServerStatus() {
  const container = $("serverStatusCard");
  if (!container) return;
  const server = state.server || {};
  container.innerHTML = `
    <div><strong>Status:</strong> ${server.running ? "Running" : "Stopped"}</div>
    <div><strong>PID:</strong> ${server.pid || "unknown"}</div>
    <div><strong>URL:</strong> ${server.url ? `<a href="${server.url}" target="_blank" rel="noreferrer">${server.url}</a>` : "n/a"}</div>
    <div><strong>Log File:</strong> ${server.log_file || "n/a"}</div>
    <div><strong>Restart Command:</strong> <code>${server.restart_command || "n/a"}</code></div>
  `;
}

function renderSessions() {
  const list = $("sessionList");
  if (!list) return;
  list.innerHTML = "";
  state.sessions.forEach((session) => {
    const node = document.createElement("div");
    node.className = "session-item";
    node.innerHTML = `
      <div>
        <strong>${session.name}</strong>
        <div class="muted">${session.retrieval_mode} · ${session.updated_at}</div>
        <div class="muted">${shortText(session.video_dir, 80)}</div>
      </div>
      <div class="launch-actions">
        <button class="ghost" data-open-session="${session.session_id}">Open</button>
        <button class="danger" data-delete-session="${session.session_id}">Delete</button>
      </div>
    `;
    list.appendChild(node);
  });

  list.querySelectorAll("[data-open-session]").forEach((button) => {
    button.addEventListener("click", async () => {
      const session = await api(`/api/editor/sessions/${button.dataset.openSession}`);
      state.currentSession = session;
      state.activeSegmentId = session.segments[0]?.segment_id ?? null;
      renderEditor();
      switchTab("editor");
    });
  });

  list.querySelectorAll("[data-delete-session]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!confirm("Are you sure you want to delete this session?")) return;
      try {
        await api(`/api/editor/sessions/${button.dataset.deleteSession}`, { method: "DELETE" });
        await refreshBootstrap();
      } catch (error) {
        setStatus("editorStatus", error.message, true);
      }
    });
  });
}

function segmentAccent(index) {
  const accents = ["var(--green)", "var(--blue)", "var(--yellow)", "var(--orange)", "var(--pink)"];
  return accents[index % accents.length];
}

function renderEditor() {
  const workspace = $("editorWorkspace");
  if (!workspace) return;

  if (!state.currentSession) {
    workspace.classList.add("hidden");
    return;
  }

  workspace.classList.remove("hidden");
  const session = state.currentSession;
  const segments = session.segments || [];
  if (state.activeSegmentId == null && segments.length) {
    state.activeSegmentId = segments[0].segment_id;
  }

  const activeSegment = segments.find((segment) => segment.segment_id === state.activeSegmentId) || segments[0];
  if (activeSegment) {
    state.activeSegmentId = activeSegment.segment_id;
    state.movementPreference = activeSegment.movement_preference || "auto";
  }

  const segmentList = $("segmentList");
  segmentList.innerHTML = "";
  segments.forEach((segment) => {
    const selected = segment.segment_id === state.activeSegmentId;
    const card = document.createElement("div");
    card.className = `segment-card ${selected ? "active" : ""}`;
    card.style.borderLeft = `4px solid ${segmentAccent(segment.segment_id)}`;
    card.innerHTML = `
      <div class="segment-header">
        <span class="segment-index">Segment ${segment.segment_id + 1}</span>
        <span class="muted">${segment.duration.toFixed(2)}s</span>
      </div>
      <p class="segment-preview">${shortText(segment.text, 170)}</p>
      <div class="segment-meta">
        <span class="meta-pill">${segment.movement_preference}</span>
        <span class="meta-pill">${segment.duration_multiplier.toFixed(2)}x</span>
        <span class="meta-pill">${segment.candidates.length} candidates</span>
      </div>
    `;
    card.addEventListener("click", () => {
      state.activeSegmentId = segment.segment_id;
      renderEditor();
    });
    segmentList.appendChild(card);
  });

  const previewVideo = $("previewVideo");
  const previewCaption = $("previewCaption");
  const assembledVideo = $("assembledVideo");
  const assembledStatus = $("assembledStatus");

  if (!activeSegment) {
    previewVideo.removeAttribute("src");
    previewCaption.textContent = "No segment selected.";
    return;
  }

  $("segmentEditorTitle").textContent = `Segment ${activeSegment.segment_id + 1} Controls`;
  $("segmentText").value = activeSegment.text || "";
  $("segmentKeywords").value = (activeSegment.extra_keywords || []).join(", ");
  $("durationMultiplier").value = activeSegment.duration_multiplier;
  $("durationMultiplierValue").textContent = `${Number(activeSegment.duration_multiplier).toFixed(2)}x`;
  $("timingBias").value = activeSegment.timing_bias;
  $("timingBiasValue").textContent = Number(activeSegment.timing_bias).toFixed(2);

  document.querySelectorAll(".idiom").forEach((button) => {
    button.classList.toggle("active", button.dataset.movement === activeSegment.movement_preference);
  });

  const selectedCandidate =
    activeSegment.candidates.find((candidate) => candidate.candidate_id === activeSegment.selected_candidate_id) ||
    activeSegment.candidates[0];

  if (selectedCandidate) {
    previewVideo.src = selectedCandidate.video_url;
    previewCaption.textContent = `${selectedCandidate.file_name} · score ${selectedCandidate.combined_score.toFixed(3)} · movement ${selectedCandidate.motion_score.toFixed(2)}`;
  } else {
    previewVideo.removeAttribute("src");
    previewCaption.textContent = "No candidate clips available for this segment.";
  }

  if (session.assembled_video_url) {
    assembledVideo.src = session.assembled_video_url;
    assembledStatus.textContent = session.assembled_video_path;
  } else {
    assembledVideo.removeAttribute("src");
    assembledStatus.textContent = "No rendered output yet.";
  }

  const candidateGrid = $("candidateGrid");
  candidateGrid.innerHTML = "";
  activeSegment.candidates.forEach((candidate) => {
    const card = document.createElement("div");
    card.className = `candidate-card ${candidate.candidate_id === activeSegment.selected_candidate_id ? "selected" : ""}`;
    card.innerHTML = `
      <img class="candidate-thumb" src="${candidate.thumbnail_url || ""}" alt="${candidate.file_name}" />
      <div class="candidate-body">
        <div class="candidate-topline">
          <strong>${candidate.file_name}</strong>
          <span class="score">${candidate.combined_score.toFixed(3)}</span>
        </div>
        <div class="muted">semantic ${candidate.similarity_score.toFixed(3)} · motion ${candidate.motion_score.toFixed(2)}</div>
        <div class="pill-row">
          ${(candidate.matched_keywords || []).map((keyword) => `<span class="keyword-chip">${keyword}</span>`).join("")}
        </div>
        <div class="launch-actions">
          <button class="primary" data-candidate-id="${candidate.candidate_id}">Use Shot</button>
        </div>
      </div>
    `;
    candidateGrid.appendChild(card);
  });

  candidateGrid.querySelectorAll("[data-candidate-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      const updated = await api(`/api/editor/sessions/${session.session_id}/segments/${activeSegment.segment_id}/select`, {
        method: "POST",
        body: JSON.stringify({ payload: { candidate_id: button.dataset.candidateId } }),
      });
      state.currentSession = updated;
      renderEditor();
    });
  });

  const suggestions = $("keywordSuggestions");
  suggestions.innerHTML = "";
  (session.global_keywords || []).slice(0, 24).forEach((keyword) => {
    const chip = document.createElement("button");
    chip.className = "keyword-chip";
    chip.textContent = keyword;
    chip.addEventListener("click", () => {
      const existing = csvList($("segmentKeywords").value);
      if (!existing.includes(keyword)) {
        $("segmentKeywords").value = [...existing, keyword].join(", ");
      }
    });
    suggestions.appendChild(chip);
  });
}

function switchTab(tabName) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}

async function refreshBootstrap() {
  const payload = await api("/api/bootstrap");
  state.settings = payload.settings;
  state.server = payload.server || {};
  state.benchmarks = payload.benchmarks;
  state.cache = payload.cache;
  state.jobs = payload.jobs;
  state.sessions = payload.sessions;

  renderSettings();
  renderServerStatus();
  renderBenchmarks();
  renderCache();
  renderJobs();
  renderSessions();
}

async function createSession() {
  setStatus("editorStatus", "Creating session...");
  try {
    const payload = {
      name: $("sessionName").value.trim(),
      benchmark: $("sessionBenchmark").value || null,
      retrieval_mode: $("sessionMode").value,
      openclip_model: $("sessionOpenclipModel").value,
      video_dir: $("sessionVideoDir").value.trim(),
      audio: $("sessionAudio").value.trim(),
      segments: $("sessionSegments").value.trim(),
      candidate_pool_size: asNumber($("sessionPool").value, 10),
      keyword_weight: asNumber($("sessionKeywordWeight").value, 0.2),
      simple_segmentation: $("sessionSimpleSegmentation").checked,
      enable_object_detection: $("sessionObjects").checked,
      enable_face_detection: $("sessionFaces").checked,
      exact_matching_mode: $("sessionExactMatching")?.checked || false,
    };
    const session = await api("/api/editor/sessions", {
      method: "POST",
      body: JSON.stringify({ payload }),
    });
    state.currentSession = session;
    state.activeSegmentId = session.segments[0]?.segment_id ?? null;
    setStatus("editorStatus", "Editing session ready.");
    await refreshBootstrap();
    renderEditor();
  } catch (error) {
    setStatus("editorStatus", error.message, true);
  }
}

async function saveSegmentSettings() {
  const session = state.currentSession;
  if (!session) return;
  const segmentId = state.activeSegmentId;
  if (segmentId == null) return;
  try {
    const updated = await api(`/api/editor/sessions/${session.session_id}/segments/${segmentId}`, {
      method: "POST",
      body: JSON.stringify({
        payload: {
          text: $("segmentText").value,
          keywords: csvList($("segmentKeywords").value),
          movement_preference: state.movementPreference,
          duration_multiplier: asNumber($("durationMultiplier").value, 1.0),
          timing_bias: asNumber($("timingBias").value, 0),
        },
      }),
    });
    state.currentSession = updated;
    renderEditor();
    await refreshBootstrap();
  } catch (error) {
    setStatus("editorStatus", error.message, true);
  }
}

async function sessionAction(url, statusMessage) {
  if (!state.currentSession) return;
  setStatus("editorStatus", statusMessage || "");
  try {
    const updated = await api(url, { method: "POST", body: JSON.stringify({ payload: {} }) });
    state.currentSession = updated;
    renderEditor();
    await refreshBootstrap();
    setStatus("editorStatus", "Done.");
  } catch (error) {
    setStatus("editorStatus", error.message, true);
  }
}

async function moveSegment(direction) {
  if (!state.currentSession || state.activeSegmentId == null) return;
  try {
    const updated = await api(`/api/editor/sessions/${state.currentSession.session_id}/segments/${state.activeSegmentId}/move`, {
      method: "POST",
      body: JSON.stringify({ payload: { direction } }),
    });
    state.currentSession = updated;
    renderEditor();
  } catch (error) {
    setStatus("editorStatus", error.message, true);
  }
}

async function uploadBenchmark() {
  const files = $("uploadBenchmarkFiles").files;
  if (!files.length) {
    setStatus("benchmarkUploadStatus", "Choose a benchmark folder first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("benchmark_number", $("uploadBenchmarkNumber").value.trim());
  formData.append("benchmark_title", $("uploadBenchmarkTitle").value.trim());

  [...files].forEach((file) => {
    formData.append("files", file);
    formData.append("relative_paths", file.webkitRelativePath || file.name);
  });

  setStatus("benchmarkUploadStatus", "Uploading benchmark...");
  try {
    const response = await fetch("/api/benchmarks/upload", { method: "POST", body: formData });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Upload failed.");
    }
    const result = await response.json();
    state.benchmarks = result.benchmarks;
    renderBenchmarks();
    setStatus("benchmarkUploadStatus", `Uploaded benchmark ${result.result.benchmark_number}.`);
  } catch (error) {
    setStatus("benchmarkUploadStatus", error.message, true);
  }
}

async function saveSettings() {
  try {
    const payload = {
      gpu_device: $("settingGpu").value.trim(),
      output: $("settingOutput").value.trim(),
      cache_dir: $("settingCache").value.trim(),
      benchmarks_dir: $("settingBenchmarksDir").value.trim(),
      server_hostname: $("settingHostname").value.trim(),
    };
    state.settings = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({ payload }),
    });
    setStatus("settingsStatus", "Settings saved.");
    await refreshBootstrap();
  } catch (error) {
    setStatus("settingsStatus", error.message, true);
  }
}

async function clearCache(action) {
  try {
    const result = await api("/api/cache/clear", {
      method: "POST",
      body: JSON.stringify({ payload: { action } }),
    });
    state.cache = result.cache;
    renderCache();
  } catch (error) {
    setStatus("settingsStatus", error.message, true);
  }
}

async function submitJob(action) {
  const payloadByAction = {
    "quick-benchmark": {
      benchmark: $("quickBenchmark").value,
      encoder: $("quickEncoder").value,
      openclip_model: $("quickOpenclipModel").value,
      verbose: $("quickVerbose").checked,
    },
    "full-pipeline": {
      benchmark: $("fullBenchmark").value || null,
      video_dir: $("fullVideoDir").value.trim(),
      audio: $("fullAudio").value.trim(),
      segments: $("fullSegments").value.trim(),
      encoder: $("fullEncoder").value,
      no_reuse: $("fullNoReuse").checked,
      greedy: $("fullGreedy").checked,
      no_windowing: $("fullNoWindowing").checked,
    },
    "openclip-grid-search": {
      benchmark: $("openclipGridBenchmark").value,
      prompt_modes: csvList($("openclipGridPrompts").value),
      frames: csvList($("openclipGridFrames").value),
      aggregations: csvList($("openclipGridAggregations").value),
    },
    "videoprism-grid-search": {
      benchmark: $("videoprismGridBenchmark").value,
      prompt_modes: csvList($("videoprismGridPrompts").value),
      frames: csvList($("videoprismGridFrames").value),
    },
    "write-a-video-grid-search": {
      benchmark: $("wavGridBenchmark").value,
      pool_sizes: csvList($("wavGridPools").value),
      keyword_weights: csvList($("wavGridKeywordWeights").value),
      prompt_modes: csvList($("wavGridPrompts").value),
    },
    "compare-all-models": {
      benchmark: $("compareBenchmark").value.trim(),
      llm_model: $("compareLLM").value.trim(),
    },
  };

  try {
    const job = await api(`/api/jobs/run/${action}`, {
      method: "POST",
      body: JSON.stringify({ payload: payloadByAction[action] }),
    });
    state.selectedJobId = job.job_id;
    await refreshJobsOnly();
    switchTab("operations");
  } catch (error) {
    setStatus("settingsStatus", error.message, true);
  }
}

async function refreshJobsOnly() {
  state.jobs = await api("/api/jobs");
  renderJobs();
}

async function refreshServerStatusOnly() {
  state.server = await api("/api/server");
  renderServerStatus();
}

async function waitForServerReturn(timeoutMs = 30000, intervalMs = 2000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      await refreshBootstrap();
      return true;
    } catch (error) {
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }
  return false;
}

async function restartServer() {
  try {
    const result = await api("/api/server/restart", {
      method: "POST",
      body: JSON.stringify({ payload: {} }),
    });
    setStatus("serverStatusMessage", result.message || "Server restart scheduled.");
    const recovered = await waitForServerReturn();
    if (recovered) {
      setStatus("serverStatusMessage", "Server restarted and UI reconnected.");
    } else {
      setStatus("serverStatusMessage", "Restart requested. Refresh this page in a few seconds if the UI does not reconnect automatically.", true);
    }
  } catch (error) {
    setStatus("serverStatusMessage", error.message, true);
  }
}

async function stopServer() {
  try {
    const result = await api("/api/server/stop", {
      method: "POST",
      body: JSON.stringify({ payload: {} }),
    });
    const restartCommand = state.server?.restart_command || "n/a";
    setStatus("serverStatusMessage", `${result.message || "Server stop scheduled."} Restart later with: ${restartCommand}`);
  } catch (error) {
    setStatus("serverStatusMessage", error.message, true);
  }
}

async function exportSegments() {
  if (!state.currentSession) return;
  window.open(`/api/editor/sessions/${state.currentSession.session_id}/segments/export`, "_blank");
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });

  $("refreshAll").addEventListener("click", refreshBootstrap);
  $("createSession").addEventListener("click", createSession);
  $("refreshSessions").addEventListener("click", refreshBootstrap);
  $("saveSegment").addEventListener("click", saveSegmentSettings);
  $("regenerateAll").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/regenerate`, "Refreshing all candidates..."));
  $("regenerateSegment").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/segments/${state.activeSegmentId}/regenerate`, "Refreshing segment candidates..."));
  $("splitSegment").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/segments/${state.activeSegmentId}/split`, "Splitting segment..."));
  $("mergeSegment").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/segments/${state.activeSegmentId}/merge`, "Merging segment..."));
  $("assembleSession").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/assemble`, "Rendering video sequence..."));
  $("exportSegments").addEventListener("click", exportSegments);
  $("moveSegmentUp").addEventListener("click", () => moveSegment("up"));
  $("moveSegmentDown").addEventListener("click", () => moveSegment("down"));
  $("uploadBenchmarkButton").addEventListener("click", uploadBenchmark);
  $("saveSettings").addEventListener("click", saveSettings);
  $("refreshServerStatus").addEventListener("click", refreshServerStatusOnly);
  $("restartServer").addEventListener("click", restartServer);
  $("stopServer").addEventListener("click", stopServer);

  $("durationMultiplier").addEventListener("input", () => {
    $("durationMultiplierValue").textContent = `${Number($("durationMultiplier").value).toFixed(2)}x`;
  });
  $("timingBias").addEventListener("input", () => {
    $("timingBiasValue").textContent = Number($("timingBias").value).toFixed(2);
  });

  document.querySelectorAll(".idiom").forEach((button) => {
    button.addEventListener("click", () => {
      state.movementPreference = button.dataset.movement;
      document.querySelectorAll(".idiom").forEach((item) => item.classList.toggle("active", item === button));
    });
  });

  document.querySelectorAll("[data-job-action]").forEach((button) => {
    button.addEventListener("click", () => submitJob(button.dataset.jobAction));
  });

  document.querySelectorAll(".cache-action").forEach((button) => {
    button.addEventListener("click", () => clearCache(button.dataset.cacheAction));
  });
}

async function initialize() {
  bindEvents();
  await refreshBootstrap();
  renderEditor();
  setInterval(refreshJobsOnly, 4000);
}

initialize();
