# V2.2-Free 指标目录与授权边界

本目录对应 **V2.2-Free**（Phase A + 瘦身 Phase B）：只落地可免费核验的来源。  
完整 V2.2 愿景中的商业商品/航运/产业链深度数据在此登记为 `needs_license` / `unavailable`，**不冒充已建成**。

运行时机器可读副本：`data/alloc_v2/maps/v22_indicator_catalog.csv`（由 `maps` / `catalog` 写出）。

## 授权分级

| license | 含义 |
| --- | --- |
| `free` | 官方免费或 AKShare/公开 API，可批量入库 |
| `needs_license` | 交易所/商业源，未获授权前 `enabled=False` |
| `unavailable` | 无稳定免费接口，保持 disabled |

## P0 免费已接入 / 可接入

| indicator_id | 主源 | freq | publish_lag | license | dataset | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| SPX/IXIC/… | akshare sina | day | T+0 | free | idx_global | V2.1 已有，不重做 |
| US/CN yields | bond_zh_us_rate | day | T+0 | free | yield_cn_us | 已有 |
| USDCNY | currency_boc_safe | day | T+0 | free | fx_usdcny | 已有 |
| SGE_GOLD | spot_golden_benchmark_sge | day | T+0 | free | gold_sge | 已有 |
| GLD_tonnes | macro_cons_gold | day | T+1 | free | gold_gld | 持仓吨位，非净流入 |
| VIX | CBOE VIX_History.csv | day | T+1 | free | vix_cboe | 官方 CSV |
| DXY | index_global_hist_em 美元指数 | day | T+0 | free | dxy | 替换不稳定 eastmoney-push |
| BRENT | futures_global_hist_em B00Y | day | T+0 | free | oil_brent | 连续合约价格 proxy |
| WTI | futures_global_hist_em CL00Y | day | T+0 | free | oil_wti | 辅助价差，非库存 |
| SGE_SILVER | spot_silver_benchmark_sge | day | T+0 | free | silver_sge | 国内白银基准 |
| SLV_tonnes | macro_cons_silver | day | T+1 | free | silver_etf | ETF 持仓吨位 |
| BTCUSDT | Binance public klines | day | T+0 | free | crypto_btc | 现货价/量；非 ETF 申赎 |
| US_REAL10Y | FRED DFII10 | day | T+1 | free | us_real_rate | 10Y TIPS 实际利率 |
| FED_WALCL | FRED WALCL | week | T+1 | free | fed_balance | 美联储资产负债表 |
| CN_M1/M2 | macro_china_money_supply | month | ~T+15d | free | cn_macro | 口径修订需保留版本 |
| CN_PMI | macro_china_pmi | month | ~T+1d | free | cn_macro | 制造业 PMI |
| CN_SOCFIN_NEW | macro_china_shrzgm | month | ~T+15d | free | cn_macro | **增量**，不是存量 |

## 商业 / 无免费批量源（disabled）

| indicator_id | 期望源 | license | dataset | 复活条件 |
| --- | --- | --- | --- | --- |
| CFTC COT | cftc.gov | needs_license | cftc_cot | 稳定免费/授权解析器 |
| LME Cu/Al 价与库存 | LME | needs_license | lme_metals | 购买数据许可 |
| COMEX 库存 | CME | needs_license | comex_inventory | 购买数据许可 |
| ICE Brent 深度/库存 | ICE | needs_license | ice_brent_depth | 购买数据许可 |
| BDI/BCI | Baltic Exchange | needs_license | baltic_freight | 购买数据许可 |
| SMM 现货 | 上海有色网 | needs_license | smm_spot | 商业授权 |
| Mysteel 库存/利润 | 我的钢铁 | needs_license | mysteel_steel | 商业授权 |
| Wind/Choice 宏观 | Wind/Choice | needs_license | wind_macro | 商业授权 |
| 碳酸锂/稀土现货深度 | SMM/交易所 | needs_license | commodity_cn_specialty | 商业授权 |
| EPFR / ICI / Z.1 | 商业/官方难批量 | unavailable | epfr_fund 等 | 保持关闭 |
| BTC ETF 申赎/资金费率 | 发行商/交易所 | unavailable | crypto_btc_flow | 稳定免费披露 |

## 商品 → 申万一级（文档级，无公司敞口）

| 全球/商品标的 | 申万一级参考 | 说明 |
| --- | --- | --- |
| 黄金/白银 | 有色金属 | 价格代理，非利好证明 |
| Brent/WTI | 石油石化 | 航空等为下游，不作自动映射 |
| 铜/铝（未接入） | 有色金属 / 公用事业 | 待授权 |
| 铁矿/螺纹（未接入） | 钢铁 / 建筑材料 | 待授权 |
| 碳酸锂/多晶硅（未接入） | 电力设备 | 待授权 |
| BTC / VIX / DXY | 无直接行业 | 风险偏好环境 |

## 验收（V2.2-Free）

1. 每个 `free` P0 有 dataset、源、PIT（`available_at`）。
2. 每个 `needs_license` / `unavailable` 在 registry 为 `enabled=False`，core 拉取不会带入。
3. 不出现假库存/假资金流；价格与持仓分字段。
4. 月频宏观不日前填、不伪装日频。
5. Dashboard 有「全球风险/商品价格代理」区，caption 标明非资金流。
