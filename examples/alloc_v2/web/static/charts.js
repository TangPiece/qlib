/** Chart.js 折线/柱与空态、热力表。 */
(function () {
  const PALETTE = ["#2563eb", "#b45309", "#157a4b", "#7c3aed", "#b42318", "#0f766e", "#374151"];
  const charts = [];

  function empty(el, msg) {
    el.innerHTML = `<div class="empty">${msg || "无数据"}</div>`;
  }

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(url + " " + res.status);
    return res.json();
  }

  function line(canvasOrBox, seriesList, opts) {
    opts = opts || {};
    const box = canvasOrBox.classList && canvasOrBox.classList.contains("chart-box")
      ? canvasOrBox
      : canvasOrBox.parentElement;
    const usable = (seriesList || []).filter((s) => s.points && s.points.length);
    if (!usable.length) {
      empty(box, (seriesList && seriesList[0] && seriesList[0].note) || "无数据");
      return null;
    }
    box.innerHTML = "<canvas></canvas>";
    const canvas = box.querySelector("canvas");
    const labels = [];
    usable.forEach((s) => s.points.forEach((p) => {
      if (p.t && !labels.includes(p.t)) labels.push(p.t);
    }));
    labels.sort();
    const datasets = usable.map((s, i) => {
      const map = Object.fromEntries(s.points.map((p) => [p.t, p.v]));
      let data = labels.map((t) => (map[t] == null ? null : map[t]));
      if (opts.normalize) {
        const first = data.find((v) => v != null && v !== 0);
        data = first ? data.map((v) => (v == null ? null : v / first)) : data;
      }
      return {
        label: s.label || s.entity_id || s.factor_id || ("序列" + (i + 1)),
        data,
        borderColor: PALETTE[i % PALETTE.length],
        backgroundColor: "transparent",
        spanGaps: true,
        pointRadius: opts.sparse ? 3 : 0,
        borderWidth: 1.6,
      };
    });
    const chart = new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
        scales: { x: { ticks: { maxTicksLimit: 8, font: { size: 10 } } } },
      },
    });
    charts.push(chart);
    return chart;
  }

  function bar(box, items, opts) {
    opts = opts || {};
    if (!items || !items.length) {
      empty(box, "无数据");
      return null;
    }
    box.innerHTML = "<canvas></canvas>";
    const canvas = box.querySelector("canvas");
    const colors = items.map((it) => (it.failed ? "#c44e52" : "#4c72b0"));
    const chart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: items.map((it) => it.label || it.entity_id),
        datasets: [{ data: items.map((it) => it.value), backgroundColor: colors }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        onClick: (_e, els) => {
          if (!opts.onClick || !els.length) return;
          opts.onClick(items[els[0].index]);
        },
      },
    });
    charts.push(chart);
    return chart;
  }

  function heatmap(box, payload) {
    if (!payload || !payload.dates || !payload.dates.length) {
      empty(box, payload && payload.note ? payload.note : "无数据");
      return;
    }
    const vals = payload.values.flat().filter((v) => v != null);
    const max = Math.max(...vals.map(Math.abs), 1e-9);
    let html = "<table><thead><tr><th></th>";
    payload.dates.forEach((d) => {
      html += `<th>${(d || "").slice(5)}</th>`;
    });
    html += "</tr></thead><tbody>";
    payload.entities.forEach((ent, i) => {
      html += `<tr data-id="${ent.id}"><th>${ent.name}</th>`;
      (payload.values[i] || []).forEach((v) => {
        if (v == null) {
          html += "<td class=\"cell\"></td>";
          return;
        }
        const t = 0.15 + 0.85 * Math.min(1, Math.abs(v) / max);
        const color = v >= 0 ? `rgba(21,122,75,${t})` : `rgba(180,35,24,${t})`;
        html += `<td class="cell" style="background:${color}">${Number(v).toFixed(2)}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    box.innerHTML = html;
  }

  async function caption(el, id) {
    if (!AllocLab._captions) {
      const data = await getJSON("/api/captions");
      AllocLab._captions = data.captions || {};
    }
    el.textContent = AllocLab._captions[id] || "";
  }

  async function seriesMany(specs, lastN) {
    const n = lastN || AllocLab.lastN();
    const out = [];
    for (const spec of specs) {
      const q = new URLSearchParams({ last_n: String(n), ...spec });
      const row = await getJSON("/api/series?" + q.toString());
      out.push(row);
    }
    return out;
  }

  AllocLab.empty = empty;
  AllocLab.getJSON = getJSON;
  AllocLab.line = line;
  AllocLab.bar = bar;
  AllocLab.heatmap = heatmap;
  AllocLab.caption = caption;
  AllocLab.seriesMany = seriesMany;
})();
