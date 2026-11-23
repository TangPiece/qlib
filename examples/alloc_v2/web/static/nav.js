/** 注入侧栏与顶栏，避免 11 页导航漂移。 */
(function () {
  const PAGES = [
    { href: "/", id: "index", label: "总览" },
    { href: "/global.html", id: "global", label: "全球资产" },
    { href: "/commodity.html", id: "commodity", label: "商品与风险" },
    { href: "/china.html", id: "china", label: "中国宏观" },
    { href: "/northbound.html", id: "northbound", label: "北向" },
    { href: "/industry.html", id: "industry", label: "行业配置" },
    { href: "/etf.html", id: "etf", label: "ETF" },
    { href: "/institution.html", id: "institution", label: "机构" },
    { href: "/signals.html", id: "signals", label: "S1–S4" },
    { href: "/backtest.html", id: "backtest", label: "回测" },
    { href: "/catalog.html", id: "catalog", label: "数据目录" },
  ];
  const WINDOWS = [
    { id: "60", lastN: 60, label: "近60" },
    { id: "120", lastN: 120, label: "近120" },
    { id: "252", lastN: 252, label: "近252" },
    { id: "all", lastN: 5000, label: "全样本" },
  ];

  window.AllocLab = window.AllocLab || {};
  AllocLab.PAGES = PAGES;
  AllocLab.WINDOWS = WINDOWS;
  AllocLab.windowId = localStorage.getItem("alloc_v2_window") || "252";
  AllocLab.lastN = function () {
    const found = WINDOWS.find((w) => w.id === AllocLab.windowId);
    return found ? found.lastN : 252;
  };

  const page = document.body.getAttribute("data-page") || "";
  const layout = document.createElement("div");
  layout.className = "layout";
  const side = document.createElement("aside");
  side.className = "side";
  side.innerHTML =
    "<h1>alloc_v2 研究站</h1><p class=\"hint\">价格 ≠ 流入 · 只读本地</p><nav></nav>";
  const nav = side.querySelector("nav");
  PAGES.forEach((p) => {
    const a = document.createElement("a");
    a.href = p.href;
    a.textContent = p.label;
    if (p.id === page) a.className = "active";
    nav.appendChild(a);
  });
  const wrap = document.createElement("div");
  wrap.className = "main";
  const top = document.createElement("header");
  top.className = "top";
  top.innerHTML =
    '<div class="kpis" id="topKpis"><span>加载中…</span></div>' +
    '<div class="evidence" title="证据标签：事实/估算/代理/信号">' +
      '<span class="tag fact">事实</span>' +
      '<span class="tag estimate">估算</span>' +
      '<span class="tag">代理</span>' +
      '<span class="tag signal">信号</span>' +
      '<span class="tag">份额≠流入</span>' +
    '</div>' +
    '<div class="windows" id="winBtns"></div>';
  const pageEl = document.querySelector("main.page");
  wrap.appendChild(top);
  if (pageEl) wrap.appendChild(pageEl);
  layout.appendChild(side);
  layout.appendChild(wrap);
  document.body.insertBefore(layout, document.body.firstChild);

  const winBox = document.getElementById("winBtns");
  WINDOWS.forEach((w) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = w.label;
    if (w.id === AllocLab.windowId) b.className = "on";
    b.addEventListener("click", () => {
      localStorage.setItem("alloc_v2_window", w.id);
      location.reload();
    });
    winBox.appendChild(b);
  });

  fetch("/api/meta")
    .then((r) => r.json())
    .then((m) => {
      const s1 = m.s1 == null ? "—" : Number(m.s1) ? "通过" : "未过";
      const s4 = m.s4_n ? m.s4_n + " 个" : "无";
      document.getElementById("topKpis").innerHTML = [
        kpi("最新日", m.last_event_date || "无数据"),
        kpi("S1", s1),
        kpi("S4", s4),
        kpi("已入库数据集", (m.n_datasets_present || 0) + " / " + (m.n_datasets_enabled || 0)),
      ].join("");
    })
    .catch(() => {
      document.getElementById("topKpis").innerHTML = "<span>无法读取 /api/meta</span>";
    });

  function kpi(label, value) {
    return `<div><span>${label}</span><b>${value}</b></div>`;
  }
})();
