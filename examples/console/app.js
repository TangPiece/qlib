(() => {
  const $ = (id) => document.getElementById(id);

  let pollTimer = null;
  let currentJobId = null;
  let officialModels = [];
  // 任务运行中时禁止操作；非滚动模型还会额外禁用滚动专用准备控件。
  let jobBusy = false;
  // 切换滚动模型时强制用 status.configs 重建左侧下拉
  let lastRollingModelId = null;

  function toast(msg, isError = false) {
    const el = $("toast");
    el.hidden = false;
    el.textContent = msg;
    el.classList.toggle("error", !!isError);
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
      el.hidden = true;
    }, 4200);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    let data = null;
    try {
      data = await res.json();
    } catch (_) {
      data = { error: `HTTP ${res.status}` };
    }
    if (!res.ok) {
      throw new Error((data && data.error) || `HTTP ${res.status}`);
    }
    return data;
  }

  function todayStamp() {
    const d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}${m}${day}`;
  }

  function selectedModel() {
    const id = ($("modelSelect") && $("modelSelect").value) || "lightgbm_rolling";
    return (
      officialModels.find((m) => m.id === id) || {
        id: "lightgbm_rolling",
        kind: "rolling",
        exp_prefix: "LightGBM_Alpha158_rolling_7y1y",
        yamls: [],
        note: "",
      }
    );
  }

  function selectedYaml() {
    const model = selectedModel();
    const name = $("yamlSelect") && $("yamlSelect").value;
    return (model.yamls || []).find((y) => y.name === name) || (model.yamls || [])[0] || null;
  }

  function isRollingModel() {
    return selectedModel().kind === "rolling";
  }

  function defaultExpName() {
    if (!isRollingModel()) {
      const model = selectedModel();
      const yaml = selectedYaml();
      const dataset = (yaml && yaml.dataset) || "default";
      return `${model.id}_${dataset}_${todayStamp()}`;
    }
    const prefix = selectedModel().exp_prefix || "rolling";
    return `${prefix}_${todayStamp()}`;
  }

  /** 官方单次训练时禁用滚动 YAML / 写入结束日，更新数据仍可用。 */
  function applyPrepRollingState() {
    const rolling = isRollingModel();
    const prepHint = $("prepRollingHint");
    if (prepHint) prepHint.hidden = rolling;
    $("configSelect").disabled = !rolling || jobBusy;
    $("endDate").disabled = !rolling || jobBusy;
    $("btnSetEnd").disabled = !rolling || jobBusy;
  }

  /** 滚动训练保留 horizon；其他模型只展示官方 YAML，并提示不能改参数。 */
  function renderModelForm() {
    const model = selectedModel();
    const rolling = model.kind === "rolling";
    $("yamlField").hidden = rolling;
    $("modelHint").hidden = rolling;
    $("horizonField").hidden = !rolling;
    if (rolling) {
      $("btnTrain").textContent = `运行 ${model.label || "滚动"}`;
      if (model.default_horizon != null) {
        $("horizon").value = String(model.default_horizon);
      }
    } else {
      $("btnTrain").textContent = "运行 qrun";
    }
    const note = $("modelNote");
    // 滚动特殊模型（HIST/TCTS 等）与官方 qrun 共用 note 提示
    if (model.note) {
      note.hidden = false;
      note.textContent = model.note;
    } else {
      note.hidden = true;
      note.textContent = "";
    }
    if (!rolling) {
      const sel = $("yamlSelect");
      const prev = sel.value;
      sel.innerHTML = "";
      (model.yamls || []).forEach((y) => {
        const opt = document.createElement("option");
        opt.value = y.name;
        opt.textContent = y.label || y.name;
        sel.appendChild(opt);
      });
      if (prev && [...sel.options].some((opt) => opt.value === prev)) {
        sel.value = prev;
      }
    } else if (lastRollingModelId !== model.id) {
      // 换滚动模型：清空 configSelect 缓存标记，等 status 重建
      delete $("configSelect").dataset.ready;
      lastRollingModelId = model.id;
    }
    applyPrepRollingState();
    if (!$("expName").dataset.touched) {
      $("expName").value = defaultExpName();
      if (!$("exportExpName").dataset.touched) {
        $("exportExpName").value = $("expName").value;
      }
    }
  }

  async function loadModels() {
    const data = await api("/api/models");
    officialModels = data.models || [];
    const sel = $("modelSelect");
    sel.innerHTML = "";
    officialModels.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label || m.id;
      sel.appendChild(opt);
    });
    if ([...sel.options].some((opt) => opt.value === "lightgbm_rolling")) {
      sel.value = "lightgbm_rolling";
    }
    renderModelForm();
  }

  function setBusy(busy) {
    jobBusy = !!busy;
    $("btnUpdateData").disabled = jobBusy;
    $("btnTrain").disabled = jobBusy;
    $("btnExport").disabled = jobBusy;
    // 滚动专用按钮还要受当前模型类型约束。
    applyPrepRollingState();
  }

  function fillConfigSelect(configs, preferredKey) {
    const sel = $("configSelect");
    const prev = preferredKey || sel.value;
    sel.innerHTML = "";
    Object.entries(configs || {}).forEach(([key, path]) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = `${key} — ${String(path).split("/").pop()}`;
      sel.appendChild(opt);
    });
    if (prev && [...sel.options].some((opt) => opt.value === prev)) {
      sel.value = prev;
    } else if (sel.options.length) {
      sel.selectedIndex = 0;
    }
    sel.dataset.ready = "1";
  }

  function renderStatus(data) {
    $("stToday").textContent = data.today || "—";
    const cal = data.calendar || {};
    if (cal.error) {
      $("stCalendar").textContent = `错误: ${cal.error}`;
    } else if (cal.first && cal.latest) {
      $("stCalendar").textContent = `${cal.first} .. ${cal.latest}`;
    } else {
      $("stCalendar").textContent = "—";
    }
    const ends = data.yaml_ends || {};
    $("stYamlEnds").textContent = [
      ends.data_handler_end || "?",
      ends.backtest_end || "?",
      ends.test_end || "?",
    ].join(" / ");

    if (data.active_job) {
      const j = data.active_job;
      $("stJob").textContent = `${j.kind} · ${j.status} · ${j.id}`;
      $("jobLog").textContent = j.log || "(等待日志…)";
      currentJobId = j.id;
      setBusy(true);
    } else if (!currentJobId) {
      $("stJob").textContent = "空闲";
      setBusy(false);
    }

    $("stMeta").textContent = `python: ${data.python || ""} · config: ${data.config_path || ""}`;

    if (data.configs && isRollingModel()) {
      if (!$("configSelect").dataset.ready) {
        fillConfigSelect(data.configs, data.config_key);
      }
    }

    if (data.today && !$("endDate").value) {
      $("endDate").value = data.today;
    }
    if (!$("expName").value) {
      $("expName").value = defaultExpName();
    }
    if (!$("exportExpName").value) {
      $("exportExpName").value = $("expName").value;
    }
  }

  async function refreshStatus() {
    // 滚动读左侧 config；官方模型读右侧所选 YAML，状态条结束日随之变化。
    const params = new URLSearchParams();
    const model = selectedModel();
    if (model.kind === "qrun") {
      params.set("model", model.id);
      const yaml = ($("yamlSelect") && $("yamlSelect").value) || "";
      if (yaml) params.set("yaml", yaml);
    } else {
      const cfg = $("configSelect").value;
      if (cfg) params.set("config", cfg);
      params.set("model", model.id);
    }
    const data = await api(`/api/status?${params.toString()}`);
    // 滚动模型切换后 configs 可能变了，强制重建
    if (isRollingModel() && data.configs) {
      if (!$("configSelect").dataset.ready) {
        fillConfigSelect(data.configs, data.config_key);
      }
    }
    renderStatus(data);
    return data;
  }

  async function refreshResults() {
    const data = await api("/api/results");
    const body = $("resultsBody");
    const rows = data.results || [];
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="6">暂无 *_positions.json / 旧 HTML</td></tr>';
      return;
    }
    body.innerHTML = rows
      .map((r) => {
        const kind = r.kind === "json" ? "JSON" : "旧HTML";
        const fav = !!r.favorited;
        // 星标切换收藏；收藏项由服务端置顶
        return `<tr class="${fav ? "is-favorited" : ""}">
        <td class="col-fav">
          <button type="button" class="btn fav ${fav ? "on" : ""}"
            data-fav-result="${escapeAttr(r.filename)}"
            data-favorited="${fav ? "1" : "0"}"
            title="${fav ? "取消收藏" : "收藏"}"
            aria-label="${fav ? "取消收藏" : "收藏"}">${fav ? "★" : "☆"}</button>
        </td>
        <td>${escapeHtml(r.exp_name)}</td>
        <td>${kind}</td>
        <td>${escapeHtml(r.mtime)}</td>
        <td>${r.size_mb} MB</td>
        <td class="actions">
          <a class="detail" href="${escapeAttr(r.url)}" target="_blank" rel="noopener">查看详情</a>
          <button type="button" class="btn danger" data-delete-result="${escapeAttr(r.filename)}" data-exp-name="${escapeAttr(r.exp_name)}">删除</button>
        </td>
      </tr>`;
      })
      .join("");
  }

  async function toggleFavorite(filename, favorited) {
    await api("/api/results/favorite", {
      method: "POST",
      body: JSON.stringify({ filename, favorited }),
    });
    toast(favorited ? "已收藏" : "已取消收藏");
    await refreshResults();
  }

  async function deleteResult(filename, expName) {
    // 连带删 MLflow 集成实验，以及列表里已经没有导出文件指向的 rolling_models。
    const ok = window.confirm(
      `删除「${expName}」？\n\n将删除该导出结果、对应的 MLflow 训练结果，以及历史列表不再引用的 rolling_models 训练数据。此操作不可恢复。`
    );
    if (!ok) return;
    const data = await api("/api/results/delete", {
      method: "POST",
      body: JSON.stringify({ filename }),
    });
    const n = (data.deleted || []).length;
    toast(`已删除 ${expName}（${n} 项）`);
    await refreshResults();
  }

  async function cleanupUnusedMlruns() {
    // 只保留历史列表引用的集成实验及其 rolling_models，其余 mlruns 实验永久删除；
    // 同时清理 .jobs 下已结束任务的日志文件。
    const ok = window.confirm(
      "清理未使用的 mlruns？\n\n将只保留历史列表中的实验及其引用的 rolling_models，其余训练结果会从磁盘永久删除；并清理已结束的任务日志。此操作不可恢复。"
    );
    if (!ok) return;
    const data = await api("/api/mlruns/cleanup", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const n = (data.deleted || []).length;
    const kept = (data.kept || []).length;
    const m = (data.deleted_logs || []).length;
    let msg;
    if (!n && !m) {
      msg = "没有需要清理的实验或任务日志";
    } else if (m) {
      msg = `已清理 ${n} 项 mlruns，${m} 个任务日志；保留 ${kept} 个实验`;
    } else {
      msg = `已清理 ${n} 项 mlruns；保留 ${kept} 个实验`;
    }
    toast(msg);
    // 任务日志区写一行摘要，便于核对（不刷完整路径列表）
    $("jobLog").textContent = msg;
    await refreshResults();
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function escapeAttr(s) {
    return escapeHtml(s).replace(/'/g, "&#39;");
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function startPoll(jobId) {
    currentJobId = jobId;
    setBusy(true);
    stopPoll();
    pollTimer = setInterval(async () => {
      try {
        const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
        $("stJob").textContent = `${job.kind} · ${job.status} · ${job.id}`;
        $("jobLog").textContent = job.log || "";
        const pre = $("jobLog");
        pre.scrollTop = pre.scrollHeight;
        if (job.status === "finished" || job.status === "failed") {
          stopPoll();
          setBusy(false);
          currentJobId = null;
          if (job.status === "finished") {
            toast(`${job.kind} 完成`);
            if (job.kind === "export") {
              refreshResults().catch(() => {});
            }
          } else {
            toast(`${job.kind} 失败: ${job.error || "unknown"}`, true);
          }
          refreshStatus().catch(() => {});
        }
      } catch (err) {
        stopPoll();
        setBusy(false);
        toast(String(err.message || err), true);
      }
    }, 2000);
  }

  async function watchJob(job) {
    if (!job || !job.id) return;
    $("jobLog").textContent = job.log || "(启动中…)";
    startPoll(job.id);
  }

  $("btnRefresh").addEventListener("click", () => {
    refreshStatus()
      .then(() => refreshResults())
      .then(() => toast("已刷新"))
      .catch((e) => toast(e.message, true));
  });

  $("btnRefreshResults").addEventListener("click", () => {
    refreshResults()
      .then(() => toast("结果列表已刷新"))
      .catch((e) => toast(e.message, true));
  });

  /** 复制任务日志全文到剪贴板，便于粘贴排查。 */
  $("btnCopyLog").addEventListener("click", async () => {
    const text = ($("jobLog").textContent || "").trim();
    if (!text || text === "尚无任务") {
      toast("暂无日志可复制", true);
      return;
    }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText($("jobLog").textContent || "");
      } else {
        const ta = document.createElement("textarea");
        ta.value = $("jobLog").textContent || "";
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      toast("日志已复制");
    } catch (e) {
      toast(e.message || "复制失败", true);
    }
  });

  $("btnCleanupMlruns").addEventListener("click", () => {
    const btn = $("btnCleanupMlruns");
    btn.disabled = true;
    cleanupUnusedMlruns()
      .catch((e) => toast(e.message, true))
      .finally(() => {
        btn.disabled = false;
      });
  });

  $("resultsBody").addEventListener("click", (ev) => {
    const favBtn = ev.target.closest("[data-fav-result]");
    if (favBtn && !favBtn.disabled) {
      favBtn.disabled = true;
      const next = favBtn.dataset.favorited !== "1";
      toggleFavorite(favBtn.dataset.favResult, next)
        .catch((e) => toast(e.message, true))
        .finally(() => {
          favBtn.disabled = false;
        });
      return;
    }
    const btn = ev.target.closest("[data-delete-result]");
    if (!btn || btn.disabled) return;
    btn.disabled = true;
    deleteResult(btn.dataset.deleteResult, btn.dataset.expName)
      .catch((e) => toast(e.message, true))
      .finally(() => {
        btn.disabled = false;
      });
  });

  $("configSelect").addEventListener("change", () => {
    refreshStatus().catch((e) => toast(e.message, true));
  });

  $("modelSelect").addEventListener("change", () => {
    // 切换模型时重置未手动改过的实验名，并重建滚动配置下拉
    delete $("expName").dataset.touched;
    delete $("exportExpName").dataset.touched;
    renderModelForm();
    refreshStatus()
      .then(() => {
        if (!$("expName").dataset.touched) {
          $("expName").value = defaultExpName();
          if (!$("exportExpName").dataset.touched) {
            $("exportExpName").value = $("expName").value;
          }
        }
      })
      .catch((e) => toast(e.message, true));
  });

  $("yamlSelect").addEventListener("change", () => {
    if (!$("expName").dataset.touched) {
      $("expName").value = defaultExpName();
      if (!$("exportExpName").dataset.touched) {
        $("exportExpName").value = $("expName").value;
      }
    }
    refreshStatus().catch((e) => toast(e.message, true));
  });

  $("expName").addEventListener("input", () => {
    $("expName").dataset.touched = "1";
  });

  $("expName").addEventListener("change", () => {
    if (!$("exportExpName").dataset.touched) {
      $("exportExpName").value = $("expName").value;
    }
  });
  $("exportExpName").addEventListener("input", () => {
    $("exportExpName").dataset.touched = "1";
  });

  $("btnSetEnd").addEventListener("click", async () => {
    try {
      const date = $("endDate").value;
      const model = selectedModel();
      const config = $("configSelect").value;
      const data = await api("/api/set-end-date", {
        method: "POST",
        body: JSON.stringify({ date, config, model: model.id }),
      });
      toast(`结束日已写入 ${data.date}`);
      await refreshStatus();
    } catch (e) {
      toast(e.message, true);
    }
  });

  $("btnUpdateData").addEventListener("click", async () => {
    try {
      const model = selectedModel();
      const payload = {
        skip_download: $("skipDownload").checked,
        force: $("forceDownload").checked,
        model: model.id,
      };
      // 滚动模型：同步当前选中的 YAML 结束日
      if (isRollingModel() && $("configSelect").value) {
        payload.configs = [$("configSelect").value];
      }
      const data = await api("/api/update-data", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      toast("已启动数据更新");
      await watchJob(data.job);
    } catch (e) {
      toast(e.message, true);
    }
  });

  $("btnTrain").addEventListener("click", async () => {
    try {
      const exp_name = $("expName").value.trim() || defaultExpName();
      $("expName").value = exp_name;
      if (!$("exportExpName").dataset.touched) {
        $("exportExpName").value = exp_name;
      }
      const model = selectedModel();
      const payload = { model: model.id, exp_name };
      if (model.kind === "qrun") {
        const yaml = $("yamlSelect").value;
        if (!yaml) {
          toast("请选择官方 YAML", true);
          return;
        }
        payload.yaml = yaml;
      } else {
        payload.config = $("configSelect").value;
        payload.horizon = Number($("horizon").value || model.default_horizon || 1);
      }
      const data = await api("/api/train", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      toast(`已启动训练 ${data.exp_name}`);
      await watchJob(data.job);
    } catch (e) {
      toast(e.message, true);
    }
  });

  $("btnExport").addEventListener("click", async () => {
    try {
      const exp_name = $("exportExpName").value.trim();
      if (!exp_name) {
        toast("请填写 exp_name", true);
        return;
      }
      const data = await api("/api/export", {
        method: "POST",
        body: JSON.stringify({ exp_name }),
      });
      toast(`已启动导出 ${data.exp_name}`);
      await watchJob(data.job);
    } catch (e) {
      toast(e.message, true);
    }
  });

  loadModels()
    .then(() => refreshStatus())
    .then((data) => {
      if (data.active_job && data.active_job.id) {
        watchJob(data.active_job);
      }
      return refreshResults();
    })
    .catch((e) => toast(e.message, true));
})();
