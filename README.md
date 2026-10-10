# INVESTMENT STRATEGY DASHBOARD
> **Hệ Thống Phân Tích & Quản Trị Danh Mục Đầu Tư Định Lượng (Quantitative Portfolio Management)**  
> **Khóa học:** Quản lý Danh mục Đầu tư (MFB025A) — Đại học Mở TP.HCM  

---

## 📌 1. Mục Tiêu Dự Án (Objective)

Dự án chuyển đổi toàn bộ nghiên cứu định lượng từ Jupyter Notebook thành một Web Dashboard chuyên nghiệp bằng **Streamlit**, sẵn sàng đưa lên GitHub và deploy trực tiếp trên **Streamlit Cloud**.

Mục tiêu cốt lõi là trình bày trọn vẹn **Quy trình Ra quyết định Đầu tư (Investment Decision-Making Workflow)** kỷ luật theo chuẩn mực của các quỹ đầu tư định lượng tổ chức (Institutional Quant Funds):
1. **Quy chuẩn nghiên cứu nghiêm ngặt:** Phân chia rạch ròi giữa giai đoạn Huấn luyện (Train: 2020–2021) và giai đoạn Kiểm định ngoài mẫu (Test: 2022).
2. **Loại bỏ triệt để Look-ahead Bias:** Mọi tín hiệu định thời điểm và trọng số đều sử dụng thông tin đã biết ở cuối phiên trước ($t-1$) để ra quyết định cho phiên $t$.
3. **Phòng chống Overfitting:** Sử dụng kỹ thuật khử nhiễu ma trận hiệp phương sai Ledoit-Wolf Shrinkage kết hợp bài toán Cực tiểu hóa biến động (Minimum Volatility), không ước lượng lợi nhuận kỳ vọng bất định.
4. **Chiến lược đầu tư có cơ sở thực chứng vững chắc:** Kết luận và khuyến nghị cuối cùng được rút ra trực tiếp từ kết quả kiểm định Out-of-Sample, trả lời rõ ràng: *Nên chọn mã nào? Phân bổ vốn ra sao? Khi nào mua? Khi nào cầm tiền?*

---

## 🏛️ 2. Khung Quy Trình 5 Phần (Updated Workflow Structure)

```
01. DATA & UNIVERSE
  └── Sàng lọc 2 tầng: 99 Mã HOSE ➔ Top 30 Thanh khoản (60D Median Value) ➔ Top 20 Eligible (RSI <= 75 & P >= SMA50)

02. STOCK SELECTION
  └── Chấm điểm 4 nhân tố (Trend 30%, Mom 30%, Risk 20%, Liq 20%) ➔ Ranking ➔ Lọc tương quan chéo (r <= 0.70) ➔ TOP 5 Stocks

03. PORTFOLIO STRATEGY
  ├── Phần 1: Cách tiếp cận bằng Shrinkage (Ledoit-Wolf Min-Vol, trần 40%)
  ├── Phần 2: Cách tiếp cận bằng Market Timing (VNINDEX SMA200, Lagged 1D)
  └── Phần 3: Mô hình Kết Hợp (Uptrend dùng Shrinkage Weights, Downtrend giữ 100% Cash)

04. BACKTEST & STRATEGY DECISION
  └── Kiểm định Out-of-Sample (2022) ➔ Đo lường 8 chỉ số hiệu suất bắt buộc ➔ So sánh trực diện với Buy & Hold Benchmark

05. TỔNG QUAN & KHUYẾN NGHỊ ĐẦU TƯ
  └── Khuyến nghị chính thức cho Hội đồng Đầu tư: Bộ nguyên tắc thi hành & Bằng chứng Out-of-Sample
```

---

## 🔍 3. Thuyết Minh Chi Tiết Từng Giai Đoạn

### Giai Đoạn 1: Dữ Liệu & Vũ Trụ Đầu Tư (01. Data & Universe)
- **Vũ trụ ban đầu:** 99 cổ phiếu niêm yết trên sàn HOSE có dữ liệu giao dịch $\ge 90\%$ số phiên trong giai đoạn Train (2020–2021).
- **Phễu sàng lọc 2 tầng (2-Tier Funnel):**
  - **Tầng 1 (Từ 99 mã ➔ Top 30 mã Thanh khoản):**
    - *Lý do:* Đảm bảo quy mô giải ngân quỹ, chống trượt giá (*slippage*) và triệt tiêu rủi ro thanh khoản.
    - *Tiêu chí:* Giá trị giao dịch trung vị 60 phiên gần nhất ($60D\ Median\ Value$). Sử dụng số Trung vị (*Median*) thay vì Trung bình (*Mean*) để loại trừ hiện tượng quay tay thanh khoản ảo.
  - **Tầng 2 (Từ Top 30 mã ➔ Top 20 mã Eligible Universe):**
    - *Tiêu chí 1 ($RSI_{14} \le 75$):* Tránh mua đuổi vào vùng quá nóng (Extreme Overbought), hạn chế rủi ro phân kỳ âm ngắn hạn.
    - *Tiêu chí 2 ($P_{close} \ge SMA_{50}$):* Loại bỏ các cổ phiếu gãy đường trung bình động 50 ngày (đã mất xu hướng tăng ngắn-trung hạn).
    - *Kết quả:* Loại đúng 10 mã (gồm các cổ phiếu lớn bị gãy SMA50: `HPG`, `TCB`, `VHM`, `VPB`, `HSG`, `NKG`, `SHB`, `DPM`, `VRE`, `VNM`), giữ lại **20 mã cổ phiếu đạt chuẩn (Eligible Universe)**.

---

### Giai Đoạn 2: Lựa Chọn Cổ Phiếu (02. Stock Selection)
- **Mô hình chấm điểm đa nhân tố (Composite Factor Score):**
  $$\text{Score} = 30\% \times \text{Trend} + 30\% \times \text{Momentum} + 20\% \times \text{Risk} + 20\% \times \text{Liquidity}$$
- **Quy tắc Correlation Filter ($r \le 0.70$):** Duyệt danh sách từ điểm cao xuống, bỏ qua các mã có tương quan lợi suất ngày $> 0.70$ với các mã đã chọn trước đó nhằm đảm bảo đa dạng hóa thực chất.
- **TOP 5 cổ phiếu được lựa chọn:** `DIG`, `GEX`, `NLG`, `VND`, `ITA`.

---

### Giai Đoạn 3: Chiến Lược Danh Mục (03. Portfolio Strategy)

Được thiết kế thành 3 chuyên đề tiếp cận chuyên sâu:

#### 🔹 Phần 1: Cách tiếp cận bằng Shrinkage (Ledoit-Wolf Min-Vol)
- Khử nhiễu ma trận hiệp phương sai mẫu bằng phương pháp Ledoit-Wolf:
  $$\hat{\Sigma}_{LW} = (1 - \delta) S + \delta F$$
- Không ước lượng lợi nhuận kỳ vọng $\mu$ để chống Overfitting. Cực tiểu hóa phương sai danh mục với ràng buộc trần $40\%$ mỗi mã:
  - `NLG`: **40.00%** (Cổ phiếu hạt nhân phòng thủ)
  - `GEX`: **23.49%**
  - `VND`: **17.97%**
  - `ITA`: **9.86%**
  - `DIG`: **8.68%**
- *Đánh giá:* Giảm độ biến động từ 52.53% xuống 50.84%, nhưng **không đủ bảo vệ vốn khi toàn thị trường sụp đổ** (vẫn lỗ -62.25% và Max DD -75.70%).

#### 🔹 Phần 2: Cách tiếp cận bằng Market Timing (VNINDEX SMA200)
- Trong Downtrend lớn, mối tương quan giữa tất cả cổ phiếu đều tiến về +1.0. Tự vệ hiệu quả nhất là **chuyển 100% Tiền Mặt (Cash)**.
- Tín hiệu trễ: $Signal_t = \mathbb{I}(VNINDEX_{t-1} > SMA200_{t-1})$.
- *Thống kê:* Năm 2022, danh mục đứng ngoài giữ tiền mặt trong **71.89% số phiên (179 phiên)**.
- *Đánh giá:* Giảm một nửa biến động (xuống 24.44%) và cắt giảm mức sụt giảm tối đa từ -79.04% xuống -26.40%.

#### 🔹 Phần 3: Mô hình Kết Hợp (Shrinkage + Market Timing)
- Khi Bull Market ($VNINDEX > SMA200$): Phân bổ vốn theo tỷ trọng tối ưu **Shrinkage Min-Vol**.
- Khi Bear Market ($VNINDEX \le SMA200$): Chuyển 100% sang **Tiền Mặt**.
- *Đánh giá:* Đạt hiệu suất cao nhất ngoài mẫu (-22.77%), độ biến động thấp nhất (23.28%), và Max Drawdown thấp nhất (-26.20%).

---

### Giai Đoạn 4: Kiểm Định Hiệu Suất (04. Backtest & Decision)

Đo lường đầy đủ 8 chỉ số tài chính bắt buộc trên tập Out-of-Sample (năm 2022):

| Chỉ số Hiệu suất | Buy & Hold Benchmark | Shrinkage Only | Market Timing Only | Combined Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **1. Total Return (Tổng Lợi Suất)** | **-68.04%** | -62.25% | -23.44% | **-22.77% (Tốt nhất)** |
| **2. CAGR (Lợi Suất Quy Năm)** | -68.62% | -62.84% | -23.77% | **-23.09% (Tốt nhất)** |
| **3. Winning Rate (Active Days)** | 47.98% | 49.19% | 49.28% | **50.72% (Tốt nhất)** |
| **4. Volatility (Độ Biến Động Năm)** | 52.53% | 50.84% | 24.44% | **23.28% (Thấp nhất)** |
| **5. Maximum Drawdown (Sụt Giảm Tối Đa)** | **-78.87%** | -75.70% | -27.76% | **-26.20% (Thấp nhất)** |
| **6. Sharpe Ratio ($R_f=3\%$)** | -1.99 | -1.75 | -1.11 | **-1.14** |
| **7. Sortino Ratio** | -2.52 | -2.22 | -1.38 | **-1.41** |
| **8. Calmar Ratio** | -0.87 | -0.83 | -0.86 | **-0.88** |
| **Market Exposure (Thời gian tham gia)** | 100.0% | 100.0% | 28.11% | **28.11% (71.89% Tiền mặt)** |

---

### Giai Đoạn 5: Tổng Quan & Khuyến Nghị (05. Recommended Strategy)

Chốt hạ kết luận cho Hội đồng Đầu tư:
1. **Chiến Lược Khuyến Nghị:** **COMBINED STRATEGY (Shrinkage Min-Vol + VNINDEX SMA200)**.
2. **Bộ Danh Mục & Tỷ Trọng:** NLG (40.00%), GEX (23.49%), VND (17.97%), ITA (9.86%), DIG (8.68%).
3. **Quy Tắc Hành Động:**
   - **INVEST:** Khi $VNINDEX_{t-1} > SMA200_{t-1}$, giải ngân 100% theo tỷ trọng Shrinkage.
   - **CASH:** Khi $VNINDEX_{t-1} \le SMA200_{t-1}$, đóng vị thế chuyển 100% về tiền mặt.
4. **Tại sao khuyến nghị?** Mang lại mức Alpha vượt trội **+45.27%** so với Buy & Hold trong Downtrend 2022, hạ Max Drawdown từ -78.87% xuống -26.20% và duy trì rủi ro thấp nhất toàn hệ thống.

---

## 💻 4. Hướng Dẫn Chạy & Deploy

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy ứng dụng Streamlit
streamlit run app.py
```

Deploy lên GitHub và Streamlit Cloud:
```bash
git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
git commit -m "feat: upgrade institutional dashboard with 5-stage workflow"
git push origin main
```
Truy cập [share.streamlit.io](https://share.streamlit.io) để kích hoạt ứng dụng trực tuyến.
