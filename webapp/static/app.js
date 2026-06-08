const state = {
  settings: {},
  server: {},
  benchmarks: [],
  cache: null,
  jobs: [],
  sessions: [],
  currentSession: null,
  bestGridSearch: null,
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
  if (!value) return [];
  return String(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function selectedValues(id) {
  const element = $(id);
  if (!element) return [];
  if (element.tagName === "SELECT") {
    const values = [...element.selectedOptions].map((option) => option.value).filter(Boolean);
    return values.length ? values : [...element.options].filter((option) => option.defaultSelected).map((option) => option.value);
  }
  return csvList(element.value);
}

function firstSelectedValue(id, fallback = "") {
  const values = selectedValues(id);
  return values.length ? values[0] : fallback;
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

  const compareSelect = $("compareBenchmark");
  if (compareSelect) {
    const current = compareSelect.value || "all";
    compareSelect.innerHTML = `<option value="all">All benchmarks</option>${benchmarkOptionsHtml(false)}`;
    compareSelect.value = [...compareSelect.options].some((option) => option.value === current) ? current : "all";
  }
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
  const sections = [
    state.cache.cache || {
      label: "Cache",
      dir: state.cache.cache_dir,
      total_size_mb: state.cache.total_size_mb,
      entries: state.cache.entries || [],
    },
    state.cache.output,
  ].filter(Boolean);

  list.innerHTML = sections.map((section) => `
    <div class="cache-item">
      <div>
        <strong>${section.label}: ${section.dir}</strong>
        <div class="muted">${section.total_size_mb} MB total · ${(section.entries || []).length} entries</div>
      </div>
    </div>
    ${(section.entries || []).map((entry) => `
      <div class="cache-item subitem">
        <div>
          <strong>${entry.name}</strong>
          <div class="muted">${entry.type} · ${entry.size_mb} MB</div>
        </div>
      </div>
    `).join("")}
  `).join("");
}

function isJobStoppable(job) {
  return ["queued", "running", "stopping"].includes(job?.status);
}

function renderJobs() {
  const list = $("jobList");
  const log = $("jobLog");
  const stopButton = $("stopSelectedJob");
  if (!list || !log) return;

  list.innerHTML = "";
  state.jobs.forEach((job) => {
    const node = document.createElement("div");
    node.className = `job-item ${job.status} ${job.job_id === state.selectedJobId ? "selected" : ""}`;
    const canStop = isJobStoppable(job);
    node.innerHTML = `
      <div>
        <strong>${job.name}</strong>
        <div class="muted">${job.status} · PID ${job.pid || "pending"} · ${job.created_at}</div>
      </div>
      <div class="launch-actions">
        <button class="ghost" data-job-id="${job.job_id}">View Log</button>
        ${canStop ? `<button class="ghost danger" data-stop-job="${job.job_id}">Stop Task</button>` : ""}
      </div>
    `;
    list.appendChild(node);
  });

  list.querySelectorAll("[data-job-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedJobId = button.dataset.jobId;
      const job = await api(`/api/jobs/${state.selectedJobId}`);
      log.textContent = job.log || "No log output yet.";
      renderJobs();
    });
  });

  list.querySelectorAll("[data-stop-job]").forEach((button) => {
    button.addEventListener("click", () => stopJob(button.dataset.stopJob));
  });

  const selected = state.jobs.find((job) => job.job_id === state.selectedJobId);
  if (selected) {
    log.textContent = selected.log || "No log output yet.";
  } else if (state.selectedJobId) {
    state.selectedJobId = null;
  }
  if (stopButton) {
    stopButton.disabled = !selected || !isJobStoppable(selected);
  }
}

function renderSettings() {
  $("settingGpu").value = state.settings.gpu_device || "";
  $("settingOutput").value = state.settings.output || "";
  $("settingCache").value = state.settings.cache_dir || "";
  $("settingBenchmarksDir").value = state.settings.benchmarks_dir || "";
  $("settingHostname").value = state.settings.server_hostname || "";
}


function compactGridValue(value) {
  if (value === null || value === undefined) return "n/a";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function renderBestGridConfig(source = state.bestGridSearch) {
  const target = $("bestGridStatus");
  if (!target) return;

  if (!source) {
    target.innerHTML = "Best grid-search settings will be applied automatically when a benchmark result exists.";
    return;
  }

  if (source.missing) {
    target.innerHTML = `<strong>Best grid-search settings:</strong> ${source.message || "No saved result found yet."}`;
    return;
  }

  const config = source.config || {};
  const fields = [
    ["Model", config.model_name],
    ["Frames", config.num_frames],
    ["Aggregation", config.aggregation],
    ["Prompt", config.prompt_mode || source.prompt_mode],
    ["Pool", config.candidate_pool_size],
    ["Keyword", config.keyword_weight],
    ["Objects", config.enable_object_detection],
    ["Faces", config.enable_face_detection],
    ["Resolution", config.resolution],
    ["Dual softmax", config.use_dual_softmax],
  ].filter(([, value]) => value !== undefined && value !== null);

  target.innerHTML = `
    <strong>Best grid-search settings:</strong>
    <span>${fields.map(([label, value]) => `${label}: ${compactGridValue(value)}`).join(" · ")}</span>
    <span class="muted">Exact match ${Number(source.exact_match_accuracy || 0).toFixed(2)}%${source.source_label ? ` · ${source.source_label}` : ""}</span>
  `;
}

async function refreshBestGridConfig() {
  const benchmark = $("sessionBenchmark")?.value;
  const retrievalMode = $("sessionMode")?.value || "writeavideo";
  if (!benchmark) {
    state.bestGridSearch = null;
    renderBestGridConfig();
    return;
  }
  try {
    state.bestGridSearch = await api(`/api/grid-search/best?benchmark=${encodeURIComponent(benchmark)}&retrieval_mode=${encodeURIComponent(retrievalMode)}`);
    renderBestGridConfig();
  } catch (error) {
    state.bestGridSearch = { missing: true, message: error.message };
    renderBestGridConfig();
  }
}

function renderServerStatus() {
  const container = $("serverStatusCard");
  if (!container) return;
  const server = state.server || {};
  const running = Boolean(server.running);
  container.innerHTML = `
    <div><strong>Status:</strong> <span class="server-pill ${running ? "running" : "stopped"}">${running ? "Running" : "Stopped"}</span></div>
    <div><strong>PID:</strong> ${server.pid || "unknown"}</div>
    <div><strong>URL:</strong> ${server.url ? `<a href="${server.url}" target="_blank" rel="noreferrer">${server.url}</a>` : "n/a"}</div>
    <div><strong>Log File:</strong> ${server.log_file || "n/a"}</div>
    <div><strong>Restart Command:</strong> <code>${server.restart_command || "n/a"}</code></div>
  `;
  const restartButton = $("restartServer");
  const stopButton = $("stopServer");
  if (restartButton) restartButton.disabled = false;
  if (stopButton) stopButton.disabled = !running;
}

function setControlVisible(fieldId, inputId, visible) {
  const field = $(fieldId);
  const input = $(inputId);
  if (!field || !input) return;
  field.hidden = !visible;
  field.classList.toggle("hidden", !visible);
  input.disabled = !visible;
}

function updateSessionModeControls() {
  const mode = $("sessionMode")?.value || "writeavideo";
  const usesOpenclip = mode === "openclip" || mode === "writeavideo";
  const usesVideoprism = mode === "videoprism";
  const usesKeywordIndex = mode === "writeavideo";
  const usesCoherenceAssignment = mode === "videoprism";
  const selectedAssignment = $("sessionAssignmentMethod")?.value || "hungarian";
  const usesCoherenceBeam = usesCoherenceAssignment && selectedAssignment === "coherence_beam";
  const selectedQueryMode = $("sessionQueryMode")?.value || "original";
  const usesContextQuery = selectedQueryMode !== "original";
  const usesLLMQuery = selectedQueryMode === "llm_expanded" || selectedQueryMode === "hybrid_llm";
  const selectedBenchmark = state.benchmarks.find((benchmark) => benchmark.number === $("sessionBenchmark")?.value);
  const hasManualSegments = Boolean($("sessionSegments")?.value.trim()) || Boolean(selectedBenchmark?.has_segments);
  const canSegmentAudio = !hasManualSegments;

  setControlVisible("sessionOpenclipModelField", "sessionOpenclipModel", usesOpenclip);
  setControlVisible("sessionVideoprismModelField", "sessionVideoprismModel", usesVideoprism);
  setControlVisible("sessionKeywordWeightField", "sessionKeywordWeight", usesKeywordIndex);
  setControlVisible("sessionObjectsField", "sessionObjects", usesKeywordIndex);
  setControlVisible("sessionFacesField", "sessionFaces", usesKeywordIndex);

  // Candidate count still controls how many alternatives are shown for every mode.
  setControlVisible("sessionPoolField", "sessionPool", true);
  setControlVisible("sessionSimpleSegmentationField", "sessionSimpleSegmentation", canSegmentAudio);
  setControlVisible("sessionAssignmentMethodField", "sessionAssignmentMethod", usesCoherenceAssignment);
  setControlVisible("sessionCoherenceTopKField", "sessionCoherenceTopK", usesCoherenceBeam);
  setControlVisible("sessionCoherenceBeamField", "sessionCoherenceBeam", usesCoherenceBeam);
  setControlVisible("sessionLambdaField", "sessionLambda", usesCoherenceBeam);
  setControlVisible("sessionNormalizeScoresField", "sessionNormalizeScores", usesCoherenceBeam);
  setControlVisible("sessionQueryModeField", "sessionQueryMode", usesVideoprism);
  setControlVisible("sessionContextWindowField", "sessionContextWindow", usesVideoprism && usesContextQuery);
  setControlVisible("sessionQueryLLMField", "sessionQueryLLM", usesVideoprism && usesLLMQuery);
  setControlVisible("sessionUseQueryCacheField", "sessionUseQueryCache", usesVideoprism && usesLLMQuery);
  setControlVisible("sessionForceRefreshExpansionsField", "sessionForceRefreshExpansions", usesVideoprism && usesLLMQuery);
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

function renderSequenceDiagnostics(session) {
  const target = $("sequenceDiagnostics");
  if (!target) return;
  const diagnostics = session?.config?.assignment_diagnostics || {};
  if (!diagnostics.assignment_method) {
    target.textContent = "Assignment diagnostics will appear after global matching.";
    return;
  }
  const transitions = diagnostics.transition_diagnostics || [];
  const query = diagnostics.query_generation || session?.config?.query_generation || {};
  const queryPreview = (query.final_queries || []).slice(0, 3).map((item) => `
    <div><strong>${Number(item.index) + 1}.</strong> ${shortText(item.final_query || item.original || "", 140)}</div>
  `).join("");
  const transitionText = transitions.slice(0, 6).map((item) => `${item.from_clip} -> ${item.to_clip}: ${Number(item.transition_coherence || 0).toFixed(3)}`).join(" · ");
  target.innerHTML = `
    <strong>Assignment:</strong> ${diagnostics.assignment_method}
    <span>Semantic ${Number(diagnostics.semantic_score || 0).toFixed(3)} · Coherence ${Number(diagnostics.coherence_score || 0).toFixed(3)} · Combined ${Number(diagnostics.combined_score || 0).toFixed(3)}</span>
    <div><strong>Query mode:</strong> ${query.query_mode || "original"} · cache ${query.cache_used ? "used" : "not used"}${query.cache_path ? ` · ${shortText(query.cache_path, 90)}` : ""}</div>
    ${queryPreview ? `<details><summary>Preview Generated Queries</summary>${queryPreview}</details>` : ""}
    ${transitionText ? `<div>${transitionText}${transitions.length > 6 ? " · ..." : ""}</div>` : ""}
  `;
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
  renderBestGridConfig(session.config?.best_grid_search || state.bestGridSearch);
  renderSequenceDiagnostics(session);
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

  let exactMatches = 0;
  let totalSegmentsWithGT = 0;

  segments.forEach((s) => {
    const sel = s.candidates.find((c) => c.candidate_id === s.selected_candidate_id) || s.candidates[0];
    if (s.ground_truth_clip) {
      totalSegmentsWithGT++;
      if (sel) {
        const predStem = sel.file_name.split(".")[0].toLowerCase().trim();
        const gtStem = s.ground_truth_clip.split(".")[0].toLowerCase().trim();
        if (predStem === gtStem) {
          exactMatches++;
        }
      }
    }
  });

  const accuracy = totalSegmentsWithGT > 0 ? (exactMatches / totalSegmentsWithGT) * 100 : null;
  const scoreBadge = $("sequenceScore");
  if (accuracy !== null) {
    scoreBadge.textContent = `Accuracy: ${accuracy.toFixed(1)}%`;
    scoreBadge.style.background = accuracy >= 80 ? "var(--green)" : accuracy >= 50 ? "var(--orange)" : "var(--red)";
    scoreBadge.classList.remove("hidden");
  } else {
    scoreBadge.classList.add("hidden");
  }

  const candidateGrid = $("candidateGrid");
  candidateGrid.innerHTML = "";
  activeSegment.candidates.forEach((candidate) => {
    const card = document.createElement("div");
    card.className = `candidate-card ${candidate.candidate_id === activeSegment.selected_candidate_id ? "selected" : ""}`;
    card.innerHTML = `
      ${candidate.is_optimal ? '<div class="optimal-badge">Optimal Match</div>' : ""}
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
      videoprism_model: $("sessionVideoprismModel").value,
      video_dir: $("sessionVideoDir").value.trim(),
      audio: $("sessionAudio").value.trim(),
      segments: $("sessionSegments").value.trim(),
      candidate_pool_size: asNumber($("sessionPool").value, 10),
      keyword_weight: asNumber($("sessionKeywordWeight").value, 0.2),
      simple_segmentation: $("sessionSimpleSegmentation").checked,
      enable_object_detection: $("sessionMode").value === "writeavideo" && $("sessionObjects").checked,
      enable_face_detection: $("sessionMode").value === "writeavideo" && $("sessionFaces").checked,
      exact_matching_mode: $("sessionExactMatching")?.checked || false,
      assignment_method: $("sessionMode").value === "videoprism" ? $("sessionAssignmentMethod").value : "hungarian",
      coherence_top_k: asNumber($("sessionCoherenceTopK")?.value, 5),
      coherence_beam_size: asNumber($("sessionCoherenceBeam")?.value, 10),
      lambda_coherence: asNumber($("sessionLambda")?.value, 0.1),
      normalize_coherence_scores: $("sessionNormalizeScores")?.checked ?? true,
      query_mode: $("sessionMode").value === "videoprism" ? ($("sessionQueryMode")?.value || "original") : "original",
      context_window_size: asNumber($("sessionContextWindow")?.value, 1),
      query_llm_model: $("sessionQueryLLM")?.value.trim() || null,
      use_query_cache: $("sessionUseQueryCache")?.checked ?? true,
      force_refresh_expansions: $("sessionForceRefreshExpansions")?.checked || false,
      use_best_grid_search: true,
    };
    const session = await api("/api/editor/sessions", {
      method: "POST",
      body: JSON.stringify({ payload }),
    });
    state.currentSession = session;
    state.activeSegmentId = session.segments[0]?.segment_id ?? null;
    state.bestGridSearch = session.config?.best_grid_search || state.bestGridSearch;
    renderBestGridConfig(state.bestGridSearch);
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


function setPipelineVisible(key, visible) {
  document.querySelectorAll(`[data-pipeline-show="${key}"]`).forEach((element) => {
    element.classList.toggle("hidden", !visible);
  });
}

function setSelectEnabled(id, enabled) {
  const element = $(id);
  if (element) element.disabled = !enabled;
}

function updatePipelineControls() {
  setPipelineVisible("quick-openclip", $("quickEncoder")?.value === "openclip");
  setPipelineVisible("full-custom-paths", !$("fullBenchmark")?.value);

  const vpAssignments = selectedValues("videoprismGridAssignments");
  const vpQueryModes = selectedValues("videoprismGridQueryModes");
  const usesCoherence = vpAssignments.includes("coherence_beam");
  const usesContext = vpQueryModes.some((mode) => ["context_window", "hybrid_llm"].includes(mode));
  const usesLlmQuery = vpQueryModes.some((mode) => ["llm_expanded", "hybrid_llm"].includes(mode));

  setPipelineVisible("videoprism-coherence", usesCoherence);
  setPipelineVisible("videoprism-context", usesContext);
  setPipelineVisible("videoprism-llm-query", usesLlmQuery);
  ["videoprismGridTopK", "videoprismGridBeams", "videoprismGridLambdas", "videoprismGridNormalizeCoherence"].forEach((id) => setSelectEnabled(id, usesCoherence));
  setSelectEnabled("videoprismGridContextWindows", usesContext);
  setSelectEnabled("videoprismGridQueryLLM", usesLlmQuery);
  setSelectEnabled("videoprismGridUseQueryCache", usesLlmQuery);
}

function bindPipelineControls() {
  [
    "quickEncoder",
    "fullBenchmark",
    "videoprismGridAssignments",
    "videoprismGridQueryModes",
  ].forEach((id) => {
    const element = $(id);
    if (element) element.addEventListener("change", updatePipelineControls);
  });
}

async function submitJob(action) {
  updatePipelineControls();
  const vpAssignments = selectedValues("videoprismGridAssignments");
  const vpQueryModes = selectedValues("videoprismGridQueryModes");
  const usesVpCoherence = vpAssignments.includes("coherence_beam");
  const usesVpContext = vpQueryModes.some((mode) => ["context_window", "hybrid_llm"].includes(mode));
  const usesVpLlmQuery = vpQueryModes.some((mode) => ["llm_expanded", "hybrid_llm"].includes(mode));

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
      models: selectedValues("openclipGridModels"),
      prompt_modes: selectedValues("openclipGridPrompts"),
      frames: selectedValues("openclipGridFrames"),
      aggregations: selectedValues("openclipGridAggregations"),
      no_windowing: $("openclipGridNoWindowing").checked,
    },
    "videoprism-grid-search": {
      benchmark: $("videoprismGridBenchmark").value,
      models: selectedValues("videoprismGridModels"),
      prompt_modes: selectedValues("videoprismGridPrompts"),
      frames: selectedValues("videoprismGridFrames"),
      resolutions: selectedValues("videoprismGridResolutions"),
      dual_softmax: selectedValues("videoprismGridDualSM"),
      assignment_methods: vpAssignments,
      coherence_top_k: usesVpCoherence ? selectedValues("videoprismGridTopK") : [],
      coherence_beam_sizes: usesVpCoherence ? selectedValues("videoprismGridBeams") : [],
      lambda_coherence_values: usesVpCoherence ? selectedValues("videoprismGridLambdas") : [],
      disable_normalize_coherence_scores: usesVpCoherence ? !$("videoprismGridNormalizeCoherence").checked : false,
      query_modes: vpQueryModes,
      context_window_sizes: usesVpContext ? selectedValues("videoprismGridContextWindows") : [],
      query_llm_model: usesVpLlmQuery ? ($("videoprismGridQueryLLM")?.value.trim() || null) : null,
      use_query_cache: usesVpLlmQuery ? $("videoprismGridUseQueryCache").checked : true,
      no_windowing: $("videoprismGridNoWindowing").checked,
    },
    "write-a-video-grid-search": {
      benchmark: $("wavGridBenchmark").value,
      models: selectedValues("wavGridModels"),
      frames: selectedValues("wavGridFrames"),
      aggregations: selectedValues("wavGridAggregations"),
      pool_sizes: selectedValues("wavGridPools"),
      keyword_weights: selectedValues("wavGridKeywordWeights"),
      prompt_modes: selectedValues("wavGridPrompts"),
      disable_object_detection: !$("wavGridObjectDetection").checked,
      disable_face_detection: !$("wavGridFaceDetection").checked,
      no_windowing: $("wavGridNoWindowing").checked,
    },
    "compare-all-models": {
      benchmark: $("compareBenchmark").value.trim(),
      llm_model: $("compareLLM").value.trim(),
      no_windowing: $("compareNoWindowing").checked,
    },
  };

  const payload = payloadByAction[action];
  if (!payload) return;
  if (Object.prototype.hasOwnProperty.call(payload, "benchmark") && !payload.benchmark) {
    setStatus("pipelineStatus", "Choose a benchmark before launching this job.", true);
    return;
  }

  try {
    setStatus("pipelineStatus", "Starting job...");
    const job = await api(`/api/jobs/run/${action}`, {
      method: "POST",
      body: JSON.stringify({ payload }),
    });
    state.selectedJobId = job.job_id;
    setStatus("pipelineStatus", `${job.name || "Job"} started. Opening Operations logs...`);
    await refreshJobsOnly();
    switchTab("operations");
  } catch (error) {
    setStatus("pipelineStatus", error.message, true);
  }
}

async function refreshJobsOnly() {
  try {
    state.jobs = await api("/api/jobs");
    renderJobs();
    setStatus("jobStatusMessage", "");
  } catch (error) {
    setStatus("jobStatusMessage", error.message, true);
  }
}

async function stopJob(jobId = state.selectedJobId) {
  if (!jobId) {
    setStatus("jobStatusMessage", "Select a running job first.", true);
    return;
  }
  const stopButton = $("stopSelectedJob");
  if (stopButton) stopButton.disabled = true;
  try {
    setStatus("jobStatusMessage", "Stopping selected job...");
    const job = await api(`/api/jobs/${jobId}/stop`, {
      method: "POST",
      body: JSON.stringify({ payload: {} }),
    });
    state.selectedJobId = job.job_id;
    await refreshJobsOnly();
    const selected = state.jobs.find((item) => item.job_id === job.job_id) || job;
    $("jobLog").textContent = selected.log || job.log || "Job stopped.";
    setStatus("jobStatusMessage", `Job ${job.job_id.slice(0, 8)} stopped.`);
  } catch (error) {
    setStatus("jobStatusMessage", error.message, true);
  } finally {
    renderJobs();
  }
}

async function refreshServerStatusOnly() {
  setStatus("serverStatusMessage", "Refreshing server status...");
  try {
    state.server = await api("/api/server");
    renderServerStatus();
    setStatus("serverStatusMessage", state.server.running ? "Server is running." : "Server is stopped.", !state.server.running);
  } catch (error) {
    setStatus("serverStatusMessage", error.message, true);
  }
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
  const restartButton = $("restartServer");
  const stopButton = $("stopServer");
  const refreshButton = $("refreshServerStatus");
  [restartButton, stopButton, refreshButton].forEach((button) => { if (button) button.disabled = true; });
  try {
    const result = await api("/api/server/restart", {
      method: "POST",
      body: JSON.stringify({ payload: {} }),
    });
    setStatus("serverStatusMessage", result.message || "Server restart scheduled.");
    const recovered = await waitForServerReturn(60000, 2000);
    if (recovered) {
      await refreshServerStatusOnly();
      setStatus("serverStatusMessage", "Server restarted and UI reconnected.");
    } else {
      setStatus("serverStatusMessage", "Restart requested. Refresh this page in a few seconds if the UI does not reconnect automatically.", true);
    }
  } catch (error) {
    setStatus("serverStatusMessage", error.message, true);
  } finally {
    [restartButton, refreshButton].forEach((button) => { if (button) button.disabled = false; });
    renderServerStatus();
  }
}

async function stopServer() {
  const restartCommand = state.server?.restart_command || "cd /data/giannis_pantrakis/video-sequencer && /home/giannis_pantrakis/miniconda3/bin/python src/webapp.py";
  const restartButton = $("restartServer");
  const stopButton = $("stopServer");
  if (restartButton) restartButton.disabled = true;
  if (stopButton) stopButton.disabled = true;
  try {
    const result = await api("/api/server/stop", {
      method: "POST",
      body: JSON.stringify({ payload: {} }),
    });
    setStatus("serverStatusMessage", `${result.message || "Server stop scheduled."} Restart later over SSH with: ${restartCommand}`);
  } catch (error) {
    setStatus("serverStatusMessage", error.message, true);
    renderServerStatus();
  }
}

async function exportSegments() {
  if (!state.currentSession) return;
  window.open(`/api/editor/sessions/${state.currentSession.session_id}/segments/export`, "_blank");
}

async function downloadVideo() {
  if (!state.currentSession || !state.currentSession.assembled_video_path) {
    alert("No rendered video available to download yet.");
    return;
  }
  const path = state.currentSession.assembled_video_path;
  const name = state.currentSession.name.replace(/\s+/g, "_") + ".mp4";
  window.open(`/api/media?path=${encodeURIComponent(path)}&download=true&filename=${encodeURIComponent(name)}`, "_blank");
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });

  $("refreshAll").addEventListener("click", refreshBootstrap);
  $("sessionMode").addEventListener("change", () => {
    updateSessionModeControls();
    refreshBestGridConfig();
  });
  $("sessionBenchmark").addEventListener("change", () => {
    updateSessionModeControls();
    refreshBestGridConfig();
  });
  $("sessionSegments").addEventListener("input", updateSessionModeControls);
  $("sessionAssignmentMethod").addEventListener("change", updateSessionModeControls);
  $("sessionQueryMode").addEventListener("change", updateSessionModeControls);
  $("createSession").addEventListener("click", createSession);
  $("refreshSessions").addEventListener("click", refreshBootstrap);
  $("saveSegment").addEventListener("click", saveSegmentSettings);
  $("regenerateAll").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/regenerate`, "Refreshing all candidates..."));
  $("regenerateSegment").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/segments/${state.activeSegmentId}/regenerate`, "Refreshing segment candidates..."));
  $("splitSegment").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/segments/${state.activeSegmentId}/split`, "Splitting segment..."));
  $("mergeSegment").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/segments/${state.activeSegmentId}/merge`, "Merging segment..."));
  $("assembleSession").addEventListener("click", () => sessionAction(`/api/editor/sessions/${state.currentSession.session_id}/assemble`, "Rendering video sequence..."));
  $("exportSegments").addEventListener("click", exportSegments);
  $("downloadVideo").addEventListener("click", downloadVideo);
  $("moveSegmentUp").addEventListener("click", () => moveSegment("up"));
  $("moveSegmentDown").addEventListener("click", () => moveSegment("down"));
  $("uploadBenchmarkButton").addEventListener("click", uploadBenchmark);
  $("saveSettings").addEventListener("click", saveSettings);
  $("refreshServerStatus").addEventListener("click", refreshServerStatusOnly);
  $("refreshJobs")?.addEventListener("click", refreshJobsOnly);
  $("stopSelectedJob")?.addEventListener("click", () => stopJob());
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

  bindPipelineControls();

  document.querySelectorAll("[data-job-action]").forEach((button) => {
    button.addEventListener("click", () => submitJob(button.dataset.jobAction));
  });

  document.querySelectorAll(".cache-action").forEach((button) => {
    button.addEventListener("click", () => clearCache(button.dataset.cacheAction));
  });
}

async function initialize() {
  bindEvents();
  updateSessionModeControls();
  updatePipelineControls();
  await refreshBootstrap();
  updateSessionModeControls();
  updatePipelineControls();
  await refreshBestGridConfig();
  renderEditor();
  setInterval(refreshJobsOnly, 4000);
}

initialize();
