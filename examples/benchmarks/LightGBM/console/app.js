(() => {
  const $ = (id) => document.getElementById(id);

  let pollTimer = null;
  let currentJobId = null;

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

  function defaultExpName() {
    return `rolling_lgb_7y1y_${todayStamp()}`;
  }

  function setBusy(busy) {
    ["btnSetEnd", "btnUpdateData", "btnTrain", "btnExport"].forEach((id) => {
      $(id).disabled = !!busy;
    });
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

    const sel = $("configSelect");
    if (data.configs && !sel.dataset.ready) {
      sel.innerHTML = "";
      Object.entries(data.configs).forEach(([key, path]) => {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = `${key} — ${path.split("/").pop()}`;
        sel.appendChild(opt);
      });
      sel.value = data.config_key || "7y1y";
      sel.dataset.ready = "1";
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
    const config = $("configSelect").value || "7y1y";
    const data = await api(`/api/status?config=${encodeURIComponent(config)}`);
    renderStatus(data);
    return data;
  }

  async function refreshResults() {
    const data = await api("/api/results");
    const body = $("resultsBody");
    const rows = data.results || [];
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="5">暂无 *_positions.json / 旧 HTML</td></tr>';
      return;
    }
    body.innerHTML = rows
      .map((r) => {
        const kind = r.kind === "json" ? "JSON" : "旧HTML";
        return `<tr>
        <td>${escapeHtml(r.exp_name)}</td>
        <td>${kind}</td>
        <td>${escapeHtml(r.mtime)}</td>
        <td>${r.size_mb} MB</td>
        <td><a class="detail" href="${escapeAttr(r.url)}" target="_blank" rel="noopener">查看详情</a></td>
      </tr>`;
      })
      .join("");
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

  $("configSelect").addEventListener("change", () => {
    refreshStatus().catch((e) => toast(e.message, true));
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
      const config = $("configSelect").value || "7y1y";
      const data = await api("/api/set-end-date", {
        method: "POST",
        body: JSON.stringify({ date, config }),
      });
      toast(`结束日已写入 ${data.date}`);
      await refreshStatus();
    } catch (e) {
      toast(e.message, true);
    }
  });

  $("btnUpdateData").addEventListener("click", async () => {
    try {
      const data = await api("/api/update-data", {
        method: "POST",
        body: JSON.stringify({
          skip_download: $("skipDownload").checked,
          force: $("forceDownload").checked,
        }),
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
      const data = await api("/api/train", {
        method: "POST",
        body: JSON.stringify({
          config: $("configSelect").value || "7y1y",
          exp_name,
          horizon: Number($("horizon").value || 1),
        }),
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

  refreshStatus()
    .then((data) => {
      if (data.active_job && data.active_job.id) {
        watchJob(data.active_job);
      }
      return refreshResults();
    })
    .catch((e) => toast(e.message, true));
})();
