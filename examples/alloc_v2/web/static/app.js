/** 按 data-page 启动各专题页。 */
document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.getAttribute("data-page") || "index";
  const boot = PAGE_BOOT[page];
  if (boot) boot().catch((err) => console.error(err));
});

const PAGE_BOOT = {
  index: bootIndex,
  global: bootGlobal,
  commodity: bootCommodity,
  china: bootChina,
  northbound: bootNorth,
  industry: bootIndustry,
  etf: bootEtf,
  institution: bootInstitution,
  signals: bootSignals,
  backtest: bootBacktest,
  catalog: bootCatalog,
};

function tag(kind, text) {
  return `<span class="tag ${kind || ""}">${text}</span>`;
}

async function bootIndex() {
  const meta = await AllocLab.getJSON("/api/meta");
  const latest = await AllocLab.getJSON("/api/industry/latest?factor_id=IndustryAllocationScore&top_n=5");
  const sig = await AllocLab.getJSON("/api/signals");
  const host = document.getElementById("overviewCards");
  const s1 = meta.s1 ? "通过" : "未过";
  host.innerHTML = [
    cardLink("/global.html", "全球环境", meta.last_event_date || "无数据", "指数 / 利率 / 黄金"),
    cardLink("/commodity.html", "风险与商品", "VIX / 原油 / BTC", "价格代理，非资金流"),
    cardLink("/industry.html", "行业 Top", (latest.items[0] && latest.items[0].name) || "无", "配置综合分"),
    cardLink("/signals.html", "S1 / S4", s1 + " / " + (meta.s4_n || 0), "研究信号，非下单"),
  ].join("");
  const series = await AllocLab.seriesMany([
    { dataset: "idx_global", entity: "CSI300", field: "close" },
    { dataset: "idx_global", entity: "SPX", field: "close" },
    { dataset: "gold_sge", entity: "SGE_GOLD", field: "close" },
  ]);
  AllocLab.line(document.getElementById("ovChart"), series, { normalize: true });
  const barBox = document.getElementById("ovBars");
  AllocLab.bar(
    barBox,
    (latest.items || []).map((it) => ({ label: it.name, value: it.value, entity_id: it.entity_id })),
    { onClick: (it) => (location.href = "/industry.html?entity=" + encodeURIComponent(it.entity_id)) }
  );
  document.getElementById("ovS1").textContent = JSON.stringify(sig.s1.components || {}, null, 0);
  await AllocLab.caption(document.getElementById("ovCap"), "global_index");
}

function cardLink(href, title, value, hint) {
  return `<a class="kpi-card" href="${href}"><div class="muted">${title}</div><div class="v">${value}</div><div class="lede" style="margin:6px 0 0">${hint}</div></a>`;
}

async function bootGlobal() {
  const n = AllocLab.lastN();
  AllocLab.line(
    document.getElementById("gIndex"),
    await AllocLab.seriesMany([
      { dataset: "idx_global", entity: "SPX", field: "close" },
      { dataset: "idx_global", entity: "IXIC", field: "close" },
      { dataset: "idx_global", entity: "CSI300", field: "close" },
      { dataset: "idx_global", entity: "HSI", field: "close" },
      { dataset: "gold_sge", entity: "SGE_GOLD", field: "close" },
    ]),
    { normalize: true }
  );
  AllocLab.line(
    document.getElementById("gYield"),
    await AllocLab.seriesMany([
      { dataset: "yield_cn_us", entity: "US10Y", field: "yield" },
      { dataset: "yield_cn_us", entity: "US2Y", field: "yield" },
      { dataset: "yield_cn_us", entity: "CN10Y", field: "yield" },
      { dataset: "yield_cn_us", entity: "CN2Y", field: "yield" },
    ])
  );
  // 同轴叠画：差 + 股端盈利收益率 + 债端 CN10Y（均为 %）
  AllocLab.line(
    document.getElementById("gEquityBondGap"),
    await AllocLab.seriesMany([
      { factor_id: "EquityBondYieldGap", entity: "CSI300_CN10Y" },
      { factor_id: "EarningsYield", entity: "CSI300" },
      { dataset: "yield_cn_us", entity: "CN10Y", field: "yield" },
    ])
  );
  AllocLab.line(
    document.getElementById("gFxGold"),
    await AllocLab.seriesMany([
      { dataset: "fx_usdcny", entity: "USDCNY", field: "fx" },
      { dataset: "gold_gld", entity: "GLD", field: "tonnes" },
    ])
  );
  await AllocLab.caption(document.getElementById("capIndex"), "global_index");
  await AllocLab.caption(document.getElementById("capYield"), "yield_cn_us");
  await AllocLab.caption(document.getElementById("capEquityBondGap"), "equity_bond_gap");
  await fillIndexList("globalList", "standard", n, [
    "idx_global", "idx_pe_csindex", "yield_cn_us", "fx_usdcny", "gold_sge", "gold_gld",
  ]);
}

async function bootCommodity() {
  AllocLab.line(
    document.getElementById("cRisk"),
    await AllocLab.seriesMany([
      { dataset: "vix_cboe", entity: "VIX", field: "close" },
      { dataset: "dxy", entity: "DXY", field: "close" },
      { dataset: "us_real_rate", entity: "US_REAL10Y", field: "yield" },
    ]),
    { normalize: true }
  );
  AllocLab.line(
    document.getElementById("cCmdty"),
    await AllocLab.seriesMany([
      { dataset: "oil_brent", entity: "BRENT", field: "close" },
      { dataset: "oil_wti", entity: "WTI", field: "close" },
      { dataset: "silver_sge", entity: "SGE_SILVER", field: "close" },
      { dataset: "crypto_btc", entity: "BTCUSDT", field: "close" },
    ]),
    { normalize: true }
  );
  AllocLab.line(
    document.getElementById("cSpread"),
    await AllocLab.seriesMany([
      { factor_id: "Brent_WTI_Spread", entity: "BRENT_WTI" },
      { factor_id: "Gold_Silver_Ratio", entity: "GOLD_SILVER" },
    ])
  );
  await AllocLab.caption(document.getElementById("capRisk"), "v22_risk_proxy");
  await AllocLab.caption(document.getElementById("capCmdty"), "v22_commodity_proxy");
  await fillIndexList("cmdtyList", "standard", AllocLab.lastN(), [
    "vix_cboe", "dxy", "oil_brent", "oil_wti", "silver_sge", "silver_etf", "crypto_btc", "us_real_rate", "fed_balance",
  ]);
}

async function bootChina() {
  AllocLab.line(
    document.getElementById("cMacro"),
    await AllocLab.seriesMany([
      { dataset: "cn_macro", entity: "M1", field: "m1_yoy" },
      { dataset: "cn_macro", entity: "M2", field: "m2_yoy" },
      { dataset: "cn_macro", entity: "PMI_MANU", field: "pmi" },
    ]),
    { sparse: true }
  );
  AllocLab.line(
    document.getElementById("cSocfin"),
    await AllocLab.seriesMany([{ dataset: "cn_macro", entity: "SOCFIN_NEW", field: "amount" }]),
    { sparse: true }
  );
  AllocLab.line(
    document.getElementById("cLpr"),
    await AllocLab.seriesMany([
      { dataset: "cn_macro", entity: "LPR1Y", field: "lpr" },
      { factor_id: "M1_M2_Spread", entity: "M1_M2" },
    ]),
    { sparse: true }
  );
  await AllocLab.caption(document.getElementById("capMacro"), "v22_cn_macro");
  await fillIndexList("chinaList", "standard", AllocLab.lastN(), ["cn_macro", "yield_cn_us", "margin"]);
}

async function bootNorth() {
  AllocLab.line(
    document.getElementById("nCore"),
    await AllocLab.seriesMany([
      { dataset: "hsgt_turnover", entity: "NORTH", field: "amount" },
      { dataset: "hsgt_hold", entity: "NORTH", field: "hold_mcap" },
    ]),
    { normalize: true }
  );
  AllocLab.line(
    document.getElementById("nNet"),
    await AllocLab.seriesMany([{ dataset: "hsgt_net", entity: "NORTH", field: "net_buy" }])
  );
  AllocLab.line(
    document.getElementById("nSouth"),
    await AllocLab.seriesMany([
      { dataset: "hsgt_turnover", entity: "SOUTH", field: "amount" },
      { dataset: "hsgt_hold", entity: "SOUTH", field: "hold_mcap" },
    ]),
    { normalize: true }
  );
  await AllocLab.caption(document.getElementById("capNorth"), "north_core");
  await AllocLab.caption(document.getElementById("capNet"), "north_net_ref");
}

async function bootIndustry() {
  const params = new URLSearchParams(location.search);
  const focus = params.get("entity") || "";
  const latest = await AllocLab.getJSON("/api/industry/latest?factor_id=IndustryAllocationScore&top_n=15");
  const turn = await AllocLab.getJSON("/api/industry/latest?factor_id=Industry_Turnover_Ratio&top_n=15");
  const heat = await AllocLab.getJSON("/api/industry/heatmap?factor_id=IndustryAllocationScore&days=60");
  AllocLab.bar(document.getElementById("indBar"), (latest.items || []).map((it) => ({
    label: it.name, value: it.value, entity_id: it.entity_id,
  })), { onClick: (it) => showIndustry(it.entity_id) });
  AllocLab.bar(document.getElementById("indTurn"), (turn.items || []).map((it) => ({
    label: it.name, value: it.value, entity_id: it.entity_id,
  })));
  AllocLab.heatmap(document.getElementById("indHeat"), heat);
  document.getElementById("indHeat").querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", () => showIndustry(tr.getAttribute("data-id")));
  });
  await AllocLab.caption(document.getElementById("capScore"), "industry_score_bar");
  await AllocLab.caption(document.getElementById("capHeat"), "industry_score_heat");
  await AllocLab.caption(document.getElementById("capTurn"), "industry_turnover");
  if (focus) showIndustry(focus);
  await fillIndexList("indList", "factor", AllocLab.lastN(), null, "IndustryAllocationScore");
}

async function showIndustry(entity) {
  document.getElementById("indFocusTitle").textContent = "行业详情：" + entity;
  const series = await AllocLab.seriesMany([
    { factor_id: "IndustryAllocationScore", entity },
    { factor_id: "Industry_Turnover_Ratio", entity },
    { factor_id: "IndustryETFFlow20", entity },
    { factor_id: "Industry_ResidualRet20", entity },
  ]);
  AllocLab.line(document.getElementById("indFocus"), series);
}

async function bootEtf() {
  const list = await AllocLab.getJSON("/api/etf");
  const q = document.getElementById("etfSearch");
  const table = document.getElementById("etfTable");
  function render(filter) {
    const text = (filter || "").toLowerCase();
    const rows = (list.items || []).filter((it) => {
      const blob = (it.etf_code + it.name + (it.theme || "")).toLowerCase();
      return !text || blob.includes(text);
    });
    table.innerHTML = "<tr><th>代码</th><th>名称</th><th>主题</th><th>行业</th><th></th></tr>" +
      rows.map((it) => `<tr class="clickable" data-code="${it.etf_code}">
        <td>${it.etf_code}</td><td>${it.name}</td><td>${it.theme || ""}</td>
        <td>${it.industry_name || it.primary_industry || ""}</td>
        <td>${it.is_cross_border ? tag("", "跨境") : ""}${it.feeder_target ? tag("estimate", "联接") : ""}</td>
      </tr>`).join("");
    table.querySelectorAll("tr.clickable").forEach((tr) => {
      tr.addEventListener("click", () => loadEtf(tr.getAttribute("data-code")));
    });
  }
  q.addEventListener("input", () => render(q.value));
  render("");
  const params = new URLSearchParams(location.search);
  loadEtf(params.get("code") || "510300");
  await AllocLab.caption(document.getElementById("capEtf"), "etf_three_510300");
}

async function loadEtf(code) {
  const detail = await AllocLab.getJSON("/api/etf?code=" + encodeURIComponent(code) + "&last_n=" + AllocLab.lastN());
  document.getElementById("etfTitle").textContent = (detail.name || code) + "（份额变化 ≠ 净流入）";
  AllocLab.line(document.getElementById("etfShares"), [detail.series.RawShares]);
  AllocLab.line(document.getElementById("etfChg"), [detail.series.Raw_Share_Change]);
  AllocLab.line(document.getElementById("etfEst"), [detail.series.Estimated_Allocation]);
}

async function bootInstitution() {
  const tabs = [
    { id: "lhb", dataset: "lhb", field: "" },
    { id: "block_trade", dataset: "block_trade", field: "" },
    { id: "margin", dataset: "margin", field: "balance" },
    { id: "fund_holding", dataset: "fund_holding", field: "" },
  ];
  const box = document.getElementById("instTabs");
  box.innerHTML = tabs.map((t) => `<button class="btn" data-id="${t.id}">${t.dataset}</button>`).join("");
  async function show(tab) {
    box.querySelectorAll("button").forEach((b) => b.classList.toggle("primary", b.dataset.id === tab.id));
    const data = await AllocLab.getJSON(
      `/api/rows?layer=standard&dataset=${tab.dataset}&field=${tab.field || ""}&limit=50`
    );
    fillTable(document.getElementById("instTable"), data.rows || []);
    document.getElementById("instNote").textContent = `共 ${data.total} 行，仅展示 ${data.rows.length}。季报 ≠ 实时持仓；龙虎榜机构席位 ≠ 已确认公募。`;
  }
  box.querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => show(tabs.find((t) => t.id === b.dataset.id)));
  });
  show(tabs[0]);
}

async function bootSignals() {
  const data = await AllocLab.getJSON("/api/signals");
  const comps = data.s1.components || {};
  const lights = Object.keys(comps).map((k) => {
    const on = comps[k];
    return `<div class="kpi-card"><div>${k}</div><div class="v" style="color:${on ? "var(--ok)" : "var(--bad)"}">${on ? "过" : "未过"}</div></div>`;
  });
  document.getElementById("s1Lights").innerHTML = lights.join("") || "<div class=\"empty\">无 S1</div>";
  document.getElementById("s1Val").textContent = data.s1.value == null ? "—" : data.s1.value;
  fillTable(document.getElementById("s2Table"), data.s2 || []);
  fillTable(document.getElementById("s4Table"), data.s4 || []);
  await AllocLab.caption(document.getElementById("capS1"), "s1");
  await AllocLab.caption(document.getElementById("capS2"), "s2");
  await AllocLab.caption(document.getElementById("capS4"), "s4");
}

async function bootBacktest() {
  const data = await AllocLab.getJSON("/api/backtest");
  const rows = data.summary || [];
  AllocLab.bar(document.getElementById("btIc"), rows.map((r) => ({
    label: r.label || r.factor_id,
    value: r.rank_ic_mean_h20,
    failed: r.status === "failed" || r.status === "factor_missing",
  })));
  AllocLab.bar(document.getElementById("btIc252"), rows.map((r) => ({
    label: r.label || r.factor_id,
    value: r.rank_ic_mean_h20_last252,
    failed: r.status === "failed" || r.status === "factor_missing",
  })));
  AllocLab.bar(document.getElementById("btLs"), rows.map((r) => ({
    label: r.label || r.factor_id,
    value: r.long_short_net_h20,
    failed: r.status === "failed" || r.status === "factor_missing",
  })));
  fillTable(document.getElementById("btYear"), data.by_year || []);
  fillTable(document.getElementById("btCorr"), data.corr || []);
  await AllocLab.caption(document.getElementById("capIc"), "backtest_ic");
  await AllocLab.caption(document.getElementById("capIc252"), "backtest_ic_last252");
  await AllocLab.caption(document.getElementById("capLs"), "backtest_ls");
}

async function bootCatalog() {
  const incoming = new URLSearchParams(location.search);
  let layer = incoming.get("layer") || "standard";
  let offset = 0;
  const q = document.getElementById("catSearch");
  if (incoming.get("q")) q.value = incoming.get("q");
  function markTabs() {
    document.querySelectorAll("button[data-layer]").forEach((b) => {
      b.classList.toggle("primary", b.getAttribute("data-layer") === layer);
    });
  }
  async function load() {
    markTabs();
    const query = encodeURIComponent(q.value || "");
    const data = await AllocLab.getJSON(`/api/index?layer=${layer}&q=${query}&offset=${offset}&limit=50`);
    const tb = document.getElementById("catTable");
    tb.innerHTML = "<tr><th>层</th><th>数据集</th><th>实体</th><th>字段</th><th>行数</th><th>区间</th></tr>" +
      (data.rows || []).map((r) => `<tr class="clickable"
        data-layer="${r.layer}" data-dataset="${r.dataset || ""}" data-entity="${r.entity_id || ""}"
        data-field="${r.field || ""}" data-factor="${r.factor_id || ""}">
        <td>${r.layer}</td><td>${r.dataset || ""}</td><td>${r.entity_id || ""} ${r.label ? "(" + r.label + ")" : ""}</td>
        <td>${r.field || r.factor_id || ""}</td><td>${r.n_rows}</td>
        <td>${(r.start || "") + (r.end ? " → " + r.end : "")}</td></tr>`).join("");
    tb.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => openSeries(tr.dataset)));
    document.getElementById("catPager").textContent = `共 ${data.total} · 偏移 ${data.offset}`;
    document.getElementById("btnPrev").disabled = offset <= 0;
    document.getElementById("btnNext").disabled = offset + 50 >= data.total;
  }
  document.querySelectorAll("button[data-layer]").forEach((b) => {
    b.addEventListener("click", () => {
      layer = b.getAttribute("data-layer");
      offset = 0;
      load();
    });
  });
  q.addEventListener("change", () => { offset = 0; load(); });
  q.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { offset = 0; load(); }
  });
  document.getElementById("btnPrev").addEventListener("click", () => { offset = Math.max(0, offset - 50); load(); });
  document.getElementById("btnNext").addEventListener("click", () => { offset += 50; load(); });
  /** 点目录行：拉单序列曲线 + 最近 50 行，不把整表塞进浏览器。 */
  async function openSeries(ds) {
    const params = new URLSearchParams({ last_n: String(AllocLab.lastN()), layer: ds.layer || layer });
    if (ds.layer === "factor" || ds.factor) {
      params.set("factor_id", ds.factor || ds.field);
      params.set("entity", ds.entity);
    } else {
      params.set("dataset", ds.dataset);
      params.set("entity", ds.entity);
      params.set("field", ds.field);
    }
    const ser = await AllocLab.getJSON("/api/series?" + params.toString());
    document.getElementById("catTitle").textContent = ser.label || "序列";
    AllocLab.line(document.getElementById("catChart"), [ser]);
    const rows = await AllocLab.getJSON("/api/rows?" + params.toString() + "&limit=50");
    fillTable(document.getElementById("catRows"), rows.rows || []);
  }
  await load();
  if (incoming.get("entity") || incoming.get("field") || incoming.get("factor_id") || incoming.get("dataset")) {
    await openSeries({
      layer,
      dataset: incoming.get("dataset") || "",
      entity: incoming.get("entity") || "",
      field: incoming.get("field") || "",
      factor: incoming.get("factor_id") || "",
    });
  }
  const reg = await AllocLab.getJSON("/api/registry");
  fillTable(document.getElementById("regTable"), reg.rows || []);
}

/** 专题页「本层全部序列」：点行进目录钻取，避免本页塞全量点。 */
async function fillIndexList(id, layer, lastN, datasets, factorId) {
  const el = document.getElementById(id);
  if (!el) return;
  const data = await AllocLab.getJSON(`/api/index?layer=${layer}&limit=200`);
  let rows = data.rows || [];
  if (datasets) rows = rows.filter((r) => datasets.includes(r.dataset));
  if (factorId) rows = rows.filter((r) => r.factor_id === factorId || r.field === factorId);
  el.innerHTML = "<tr><th>数据集</th><th>实体</th><th>字段</th><th>行数</th></tr>" +
    (rows.map((r) => {
      const href = `/catalog.html?layer=${encodeURIComponent(layer)}` +
        `&dataset=${encodeURIComponent(r.dataset || "")}` +
        `&entity=${encodeURIComponent(r.entity_id || "")}` +
        `&field=${encodeURIComponent(r.field || "")}` +
        `&factor_id=${encodeURIComponent(r.factor_id || "")}`;
      return `<tr class="clickable" onclick="location.href='${href}'"><td>${r.dataset || r.factor_id}</td><td>${r.entity_id}</td><td>${r.field}</td><td>${r.n_rows}</td></tr>`;
    }).join("") || "<tr><td colspan=\"4\">无序列</td></tr>");
}

function fillTable(el, rows) {
  if (!el) return;
  if (!rows.length) {
    el.innerHTML = "<tr><td>无数据</td></tr>";
    return;
  }
  const cols = Object.keys(rows[0]);
  el.innerHTML = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>" +
    rows.map((r) => "<tr>" + cols.map((c) => `<td>${fmtCell(r[c])}</td>`).join("") + "</tr>").join("");
}

function fmtCell(v) {
  if (v == null) return "";
  if (typeof v === "object") return JSON.stringify(v);
  const s = String(v);
  return s.length > 80 ? s.slice(0, 77) + "…" : s;
}
