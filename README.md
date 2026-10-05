# INVESTMENT STRATEGY DASHBOARD
> **Hệ Thống Phân Tích & Quản Trị Danh Mục Đầu Tư Định Lượng (Quantitative Portfolio Management)**  
> **Khóa học:** Quản lý Danh mục Đầu tư (MFB025A) — Đại học Mở TP.HCM  

---

## 📌 1. Mục Tiêu Dự Án (Objective)

Dự án chuyển đổi toàn bộ nghiên cứu định lượng từ Jupyter Notebook thành một Web Dashboard chuyên nghiệp bằng **Streamlit**, sẵn sàng đưa lên GitHub và deploy trực tiếp trên **Streamlit Cloud**.

Mục tiêu cốt lõi không chỉ dừng lại ở việc trực quan hóa kết quả backtest, mà thể hiện rõ nét **Quy trình Ra quyết định Đầu tư (Investment Decision-Making Process)** kỷ luật theo chuẩn mực các quỹ định lượng (Institutional Quant Fund):
1. **Tuân thủ quy chuẩn nghiên cứu nghiêm ngặt:** Phân chia rõ ràng giữa giai đoạn Huấn luyện (Train: 2020–2021) và giai đoạn Kiểm định ngoài mẫu (Test: 2022).
2. **Loại bỏ hoàn toàn Look-ahead Bias:** Mọi tín hiệu giao dịch đều dùng dữ liệu trễ ($t-1$) để áp dụng cho ngày $t$.
3. **Chống Overfitting (Chống quá mức khớp mẫu):** Lựa chọn mô hình tối ưu rủi ro tối thiểu (Minimum Volatility) qua hiệp phương sai Ledoit-Wolf Shrinkage, không cố gắng dự báo lợi nhuận kỳ vọng bất định.
4. **Đưa ra Chiến lược Đầu tư Khuyến nghị (Final Investment Strategy):** Có cơ sở vững chắc từ kết quả Out-of-Sample, trả lời rõ: *Chọn mã nào? Phân bổ bao nhiêu? Khi nào mua? Khi nào cầm tiền?*

---

## 🏛️ 2. Khung Phương Pháp 4 Giai Đoạn (4-Stage Framework)

```
STAGE 1 — DATA & STOCK UNIVERSE
  └── Market Data ➔ Data Cleaning ➔ Train/Test Split ➔ Technical Indicators ➔ Liquidity Filter ➔ Eligible Universe

STAGE 2 — STOCK SELECTION
  └── 4-Factor Scoring (Trend, Momentum, Risk, Liquidity) ➔ Ranking ➔ Correlation Filter (r <= 0.70) ➔ TOP 5 Stocks

STAGE 3 — PORTFOLIO STRATEGY
  ├── How to allocate? ➔ Equal Weight (1/N) vs. Ledoit-Wolf Shrinkage Min-Vol (Cap 40%)
  └── When to invest?  ➔ Market Timing: VNINDEX > SMA200 (Invest) vs. VNINDEX <= SMA200 (Cash)

STAGE 4 — BACKTEST & STRATEGY DECISION
  └── Out-of-Sample Verification (2022) ➔ 8 Performance Metrics ➔ Comparative Analysis ➔ Final Recommendation
```

---

## 🔍 3. Chi Tiết Từng Giai Đoạn

### Giai Đoạn 1: Dữ Liệu & Vũ Trụ Đầu Tư (Stage 1 — Data & Universe)
- **Nguồn dữ liệu:** Dữ liệu giá giao dịch sàn HOSE giai đoạn 2020–2023 (`HOSE_2020_2023_in.csv`).
- **Phân tách dữ liệu:**
  - **Tập Train (In-Sample):** 02/01/2020 – 31/12/2021 (502 phiên giao dịch).
  - **Tập Test (Out-of-Sample):** 04/01/2022 – 30/12/2022 (249 phiên giao dịch).
  - Tách riêng chỉ số `VNINDEX` làm dữ liệu tham chiếu thị trường.
- **Làm sạch:** Giữ các mã có dữ liệu $\ge 90\%$ số phiên, thay thế giá khuyết thiếu bằng forward-fill.
- **Bộ chỉ báo kỹ thuật tính trên Train:**
  - `px_vs_sma50`: Giá so với SMA50 ($P_t / SMA_{50} - 1$).
  - `sma50_vs_sma200`: Đường SMA50 so với SMA200 ($SMA_{50} / SMA_{200} - 1$).
  - `mom_6m`: Động lượng 6 tháng loại trừ tháng gần nhất ($P_{t-21} / P_{t-126} - 1$).
  - `rsi14`: Chỉ số sức mạnh tương đối Wilder's RSI 14 phiên.
  - `vol_daily`: Độ lệch chuẩn lợi suất ngày (biến động lịch sử).
  - `liq_value`: Giá trị giao dịch trung vị 60 phiên gần nhất ($Median(Close \times Volume)$).
- **Bộ lọc loại trừ:** Lấy Top 30 mã thanh khoản cao nhất, sau đó loại bỏ các mã có $RSI_{14} > 75$ (quá mua cực đoan) hoặc $Giá < SMA_{50}$ (mất xu hướng tăng). Giữ lại **20 mã đủ điều kiện (Eligible Universe)**.

---

### Giai Đoạn 2: Lựa Chọn Cổ Phiếu (Stage 2 — Stock Selection)
Mô hình chấm điểm đa nhân tố (Composite Factor Score):
$$\text{Score} = 30\% \times \text{Trend} + 30\% \times \text{Momentum} + 20\% \times \text{Risk} + 20\% \times \text{Liquidity}$$

- **Phân vị xếp hạng (Percentile Rank):**
  - $\text{Trend} = (\text{Rank}(px\_vs\_sma50) + \text{Rank}(sma50\_vs\_sma200)) / 2$
  - $\text{Momentum} = (\text{Rank}(mom\_6m) + \text{Rank}(-|RSI_{14} - 60|)) / 2$
  - $\text{Risk} = \text{Rank}(-vol\_daily)$ *(Ưu tiên biến động thấp)*
  - $\text{Liquidity} = \text{Rank}(liq\_value)$ *(Ưu tiên thanh khoản cao)*
- **Bộ lọc tương quan (Correlation Filter):** Duyệt danh sách từ điểm cao xuống thấp, chỉ chọn các mã có hệ số tương quan chéo $r \le 0.70$ với các mã đã chọn trước đó nhằm đảm bảo đa dạng hóa thực chất.
- **Kết quả TOP 5 cổ phiếu được lựa chọn:** `DIG`, `GEX`, `NLG`, `VND`, `ITA`.

---

### Giai Đoạn 3: Chiến Lược Danh Mục (Stage 3 — Portfolio Strategy)
Hệ thống so sánh 4 phương pháp quản lý danh mục:

1. **Buy & Hold Benchmark (Equal Weight):** Top 5 chia đều 20% mỗi mã, nắm giữ thụ động 100% thời gian.
2. **Shrinkage Only:** Tối ưu hóa ma trận hiệp phương sai bằng phương pháp **Ledoit-Wolf Shrinkage** để tìm danh mục có biến động thấp nhất (**Minimum Volatility**), áp đặt trần tỷ trọng $40\%$ cho mỗi mã:
   - `NLG`: **40.00%**
   - `GEX`: **23.49%**
   - `VND`: **17.97%**
   - `ITA`: **9.86%**
   - `DIG`: **8.68%**
3. **Market Timing Only:** Sử dụng tín hiệu kỹ thuật $VNINDEX > SMA200$ (Lagged 1 phiên để triệt tiêu Look-ahead bias):
   - $VNINDEX_{t-1} > SMA200_{t-1}$ ➔ **INVEST (Đầu tư 100% vào Top 5 chia đều 20%)**
   - $VNINDEX_{t-1} \le SMA200_{t-1}$ ➔ **CASH (Đứng ngoài giữ 100% tiền mặt)**
4. **Combined Strategy (Shrinkage + Market Timing):**
   - Thị trường Uptrend ($VNINDEX > SMA200$) ➔ Phân bổ vốn theo tỷ trọng **Shrinkage Min-Vol**.
   - Thị trường Downtrend ($VNINDEX \le SMA200$) ➔ Chuyển toàn bộ **100% về tiền mặt**.

---

### Giai Đoạn 4: Kiểm Định Hiệu Suất (Stage 4 — Backtest & Metrics)

Bắt buộc đo lường 8 chỉ số tài chính trên tập kiểm định Out-of-Sample (năm 2022 - Downtrend khốc liệt):

| Chỉ số Hiệu suất | Ý nghĩa Tài chính & Cách tính |
| :--- | :--- |
| **1. Return (Tổng lợi suất)** | $\frac{V_{end}}{V_{start}} - 1$ (Hiệu suất lũy kế cả chu kỳ) |
| **2. CAGR (Lợi suất quy năm)** | $(1 + Return)^{252 / N} - 1$ |
| **3. Winning Rate (Active Win Rate)** | Tỷ lệ số phiên tăng giá trên tổng số phiên tham gia thị trường ($r_t > 0 / r_t \ne 0$) |
| **4. Volatility (Độ biến động năm)** | $\sigma_{daily} \times \sqrt{252}$ (Độ rủi ro dao động giá) |
| **5. Maximum Drawdown (Max DD)** | $\min \left(\frac{Equity_t}{\max_{\tau \le t} Equity_\tau} - 1\right)$ (Mức sụt giảm tối đa từ đỉnh) |
| **6. Sharpe Ratio ($R_f = 3\%$)** | $\frac{\bar{r} \times 252 - R_f}{\sigma_{annual}}$ (Hiệu suất điều chỉnh theo tổng rủi ro) |
| **7. Sortino Ratio** | $\frac{\bar{r} \times 252 - R_f}{\sigma_{downside}}$ (Hiệu suất điều chỉnh theo rủi ro giảm giá) |
| **8. Calmar Ratio** | $\frac{CAGR}{\|Max\ Drawdown\|}$ (Tỷ suất sinh lời trên mỗi đơn vị sụt giảm) |

---

## 📊 Bảng So Sánh Hiệu Suất Ngoài Mẫu (Out-of-Sample Test 2022)

| Chỉ số | Buy & Hold (Benchmark) | Shrinkage Only | Market Timing Only | Combined Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng Lợi Suất (Return)** | **-68.04%** | -62.25% | -23.44% | **-22.77%** |
| **Lợi Suất Quy Năm (CAGR)** | -68.62% | -62.84% | -23.77% | **-23.09%** |
| **Tỷ Lệ Thắng (Active Win Rate)** | 47.98% | 49.19% | 49.28% | **50.72%** |
| **Độ Biến Động (Volatility)** | 52.53% | 50.84% | 24.44% | **23.28% (Thấp nhất)** |
| **Sụt Giảm Tối Đa (Max DD)** | **-78.87%** | -75.70% | -27.76% | **-26.20% (Thấp nhất)** |
| **Sharpe Ratio ($R_f=3\%$)** | -1.99 | -1.75 | -1.11 | **-1.14** |
| **Sortino Ratio** | -2.52 | -2.22 | -1.38 | **-1.41** |
| **Calmar Ratio** | -0.87 | -0.83 | -0.86 | **-0.88** |
| **Thời Gian Tham Gia (Exposure)** | 100.0% | 100.0% | 28.11% | **28.11% (71.89% Tiền mặt)** |

---

## 🏆 KẾT LUẬN CHIẾN LƯỢC ĐẦU TƯ (FINAL INVESTMENT STRATEGY)

Từ kết quả kiểm định độc lập trên dữ liệu kiểm định ngoài mẫu năm 2022 (Downtrend lịch sử của thị trường chứng khoán Việt Nam), Hội đồng Đầu tư đưa ra khuyến nghị chính thức:

### 1. Chiến Lược Được Khuyến Nghị
👉 **COMBINED STRATEGY (Ledoit-Wolf Shrinkage Min-Vol + VNINDEX SMA200 Market Timing)**

### 2. Các Cổ Phiếu Được Lựa Chọn & Tỷ Trọng Phân Bổ Vốn
- **NLG (Nam Long):** **40.00%** *(Cổ phiếu hạt nhân phòng thủ, biến động thấp nhất)*
- **GEX (Gelex):** **23.49%**
- **VND (VNDIRECT):** **17.97%**
- **ITA (Tân Tạo):** **9.86%**
- **DIG (DIC Corp):** **8.68%** *(Cổ phiếu beta cao, được kiểm soát tỷ trọng thấp)*

### 3. Quy Tắc Hành Động (Execution Rules)
- **Khi nào INVEST?** Khi giá đóng cửa VNINDEX phiên trước vượt đường trung bình động 200 ngày ($VNINDEX_{t-1} > SMA200_{t-1}$), giải ngân 100% danh mục theo bộ tỷ trọng Shrinkage.
- **Khi nào CASH?** Khi $VNINDEX_{t-1} \le SMA200_{t-1}$, đóng toàn bộ vị thế cổ phiếu, chuyển 100% tài sản về tiền mặt hoặc chứng chỉ tiền gửi ngắn hạn.

### 4. Tại Sao Lựa Chọn Chiến Lược Này?
1. **Khắc phục nhược điểm chí mạng của Buy & Hold:** Trong thị trường suy thoái, Buy & Hold chịu mức giảm khủng khiếp **-78.87%**, làm bốc hơi toàn bộ thành quả nhiều năm trước đó.
2. **Hiệu ứng phòng vệ vượt trội của Market Timing:** Giúp danh mục đứng ngoài thị trường an toàn trong **71.89% số phiên** năm 2022.
3. **Ưu thế tối ưu hóa của Shrinkage:** Giảm độ biến động xuống mức thấp nhất hệ thống (**23.28%**) và sụt giảm tối đa chỉ **-26.20%**, mang lại mức Alpha vượt trội **+45.27%** so với Benchmark Buy & Hold.

---

## 💻 4. Hướng Dẫn Cài Đặt & Chạy Ứng Dụng (Local Run)

### Yêu cầu môi trường
- Python 3.9 trở lên (Khuyến nghị Python 3.10 hoặc 3.11).

### Bước 1: Clone Repository
```bash
git clone https://github.com/<your-username>/investment-strategy-dashboard.git
cd investment-strategy-dashboard
```

### Bước 2: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi chạy Streamlit Dashboard
```bash
streamlit run app.py
```
Sau khi chạy, ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`.

---

## 🚀 5. Hướng Dẫn Deploy Lên Streamlit Cloud & GitHub

1. **Khởi tạo Git và Commit mã nguồn:**
   ```bash
   git init
   git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
   git commit -m "feat: complete investment strategy streamlit dashboard"
   ```
2. **Tạo Repository mới trên GitHub:**
   - Đặt tên repository (ví dụ: `investment-strategy-dashboard`).
   - Đặt ở chế độ Public.
3. **Đẩy mã nguồn lên GitHub:**
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/investment-strategy-dashboard.git
   git push -u origin main
   ```
4. **Deploy trên Streamlit Community Cloud:**
   - Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub.
   - Bấm **"New app"**.
   - Chọn Repository: `<your-username>/investment-strategy-dashboard`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - Bấm **"Deploy!"**.
   - Sau 1–2 phút, ứng dụng của bạn sẽ hoạt động trực tiếp trên Internet với đường dẫn công khai.

---

## 📂 6. Cấu Trúc Dự Án (Repository Structure)

```text
├── app.py                   # Mã nguồn chính Streamlit Web Application (Full 4 Stages & Dashboard)
├── requirements.txt         # Danh sách thư viện Python cần thiết
├── README.md                # Tài liệu thuyết minh chi tiết phương pháp & hướng dẫn deploy
└── HOSE_2020_2023_in.csv    # Dữ liệu giá cổ phiếu sàn HOSE và chỉ số VNINDEX (2020-2023)
```

---

*Học viên thực hiện:* Nhóm Nghiên Cứu Định Lượng — Lớp MFB025A  
*Trường:* Trường Đại học Mở Thành phố Hồ Chí Minh (Open University)
