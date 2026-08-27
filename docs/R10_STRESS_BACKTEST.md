# AlphaPilot R10-MAX 0.5% 壓力回測

這個流程用來檢驗 **LOCKED / Forward Live Validation** 的 R10-MAX 0.5% 在歷史極端行情中的承受力，不用壓力樣本重新調參。

## 鎖定 Portfolio Layer

- 初始 NAV：NT$1,300,000，共同資金池，不再使用 62/38 固定資金桶。
- R7 新訊號：22% × 當下 Portfolio NAV。
- R0.5 新訊號：20% × 當下 Portfolio NAV。
- 最大 5 檔。
- 同一股票跨策略合併後最高 25% NAV。
- 正常總曝險最高 95% NAV。
- DD throttle：DD <= -6% 新倉倍率 85%；DD <= -9% 為 45%；DD <= -15% 為 40%。
- DD <= -14%：T+1 將總曝險主動降至約 50%，停止新倉 10 交易日，防守冷卻 15 交易日。
- 買進流動性：單筆股數 <= T 日可知 20D 平均成交量 2%。
- 股數：足買一張時只允許 1,000 股倍數；只有一張本身超過目標資金才允許整數股零股。
- 現金不得為負，不借款、不槓桿、不事後補資金。

## Strict T -> T+1

- 所有選股、出場、股數、資金與買進限價只用 T 日收盤前資訊。
- R7 買價：T 收盤約 ×0.98，向下合法 Tick。
- R0.5 買價：T 收盤 ×0.995，向下合法 Tick。
- T+1 若 Open <= Limit，以 Open 成交；否則 Low <= Limit 才以 Limit 成交；沒有碰價就取消且不追。
- 賣出由 T 日決定，T+1 Open 後以 0.5% 不利滑價模擬，再向下合法 Tick。
- 手續費買賣各 0.0855%；賣出證交稅 0.3%。

## 壓力情境

### `gfc2008`

- Warm-up：2007。
- 正式區間：2008-01-02 ~ 2009-12-31，涵蓋金融海嘯與後續復甦。
- **只能標示為 `PARTIAL_R10_R7_ONLY`。**
- 原因：TWSE 個股三大法人日資料 T86 沒有 2008 的完整官方日資料；R0.5 需要 Foreign3D / Foreign10D / Trust5D，因此禁止用虛構資料補齊。
- 這個情境只跑 R7 原始訊號 + R10 共用資金池 / 曝險 / DD / T+1 執行層。

### `covid2020`

- Warm-up：2019。
- 正式區間：2020 全年。
- `FULL_R10`：R7 + R0.5 + Portfolio Layer。

### `bear2022`

- Warm-up：2021。
- 正式區間：2022 全年。
- `FULL_R10`。

### `validation2021_2025`

用來驗證重建引擎是否能重現鎖定正式樣本。鎖定基準：

- 期末資產：NT$9,888,538
- CAGR：50.19%
- Max DD：-12.26%
- 完整交易：241 筆

**如果 regression validation 與正式基準偏差過大，新壓力回測只能標示 Research/Reconstruction，不得直接稱為正式 R10 等價結果。**

## 執行

GitHub Actions：`AlphaPilot R10-MAX Stress Backtest`

可選：`all`、`gfc2008`、`covid2020`、`bear2022`、`validation2021_2025`。

完整結果會上傳 Actions artifact；精簡 `SUMMARY.md` 與 `summary.json` 會回寫 `stress_results/latest/`，方便 ChatGPT 直接讀取。

## 不可做的事

- 不因 2008 / 2020 壓力結果修改 R10 參數後再說這是 OOS。
- 不把 2008 的 R7-only 結果包裝成完整 R10。
- 不使用 T+1 盤中資訊回頭改 T 日訂單。
- 不把沒碰到的限價單算成交。
- 不使用分數股、不使用負現金。
