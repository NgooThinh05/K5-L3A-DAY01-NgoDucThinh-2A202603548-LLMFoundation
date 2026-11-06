# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM


- **MSSV:2A202603548**
- **Họ và tên:Ngô Đức Thịnh**
- **SEED = 1947** 

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

**Bảng 1: Các chỉ số phân loại (Chất lượng mô hình)**

| Mô hình | Chỉ số 1 | Chỉ số 2 | Chỉ số 3 | Chỉ số 4 (F1) |
| :--- | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg** (tự code) | 0.989 | 0.784 | 0.874 | 0.889 |
| **2. BoW + MLP** (deep learning) | 0.980 | 0.874 | 0.924 | 0.919 |
| **3a. Encoder (MiniLM) + kNN** | 0.904 | 0.928 | 0.916 | 0.947 |
| **3b. Encoder (MiniLM) + LogReg** | 0.981 | 0.919 | 0.949 | 0.947 |
| **3c. Encoder (MiniLM) + MLP head** | 0.981 | 0.919 | 0.949 | 0.947 |
| **4a. LLM nhỏ zero-shot** | 1.000 | 0.950 | 0.974 | 0.974 |
| **4b. LLM lớn zero-shot** | 1.000 | 0.950 | 0.974 | 0.974 |
| **4c. LLM nhỏ few-shot** | 1.000 | 0.900 | 0.947 | 0.947 |

**Bảng 2: Các chỉ số hiệu năng (Tốc độ & Chi phí)**

| Mô hình | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg** | 4.394 | 18,098.12 | 0.057 | 0.108 | $0.000 |
| **2. BoW + MLP** | 10.600 | 26,020.05 | 0.305 | 2.555 | $0.000 |
| **3a. Encoder + kNN** | 100.141 | 29.15 | 51.691 | 70.433 | $0.000 |
| **3b. Encoder + LogReg** | 100.614 | 46.03 | 29.534 | 46.631 | $0.000 |
| **3c. Encoder + MLP head** | 101.653 | 44.34 | 28.071 | 36.779 | $0.000 |
| **4a. LLM nhỏ zero-shot** | 0.000 | 1.53 | 565.877 | 1239.503 | $0.019 |
| **4b. LLM lớn zero-shot** | 0.000 | 1.45 | 526.076 | 1424.687 | $0.044 |
| **4c. LLM nhỏ few-shot** | 0.000 | 1.34 | 516.859 | 2097.892 | $0.058 |

## 3. Dự đoán và thực tế

**1. Dự đoán lệch nhất: Sức chịu đựng của BoW trên tập 50 case khó (hard_eval)**
* **Đoán:** Mô hình Bag-of-Words (BoW) sẽ đạt độ chính xác tương đối, khoảng ~55% F1.
* **Thực tế:** F1 sụt giảm nghiêm trọng, chỉ còn 33% - 36%.
* **Vì sao lệch:** Đã đánh giá thấp sức tàn phá của các nhóm case khó. Không lường trước được việc mô hình BoW sẽ bị "mù" và sụp đổ hoàn toàn khi gặp các thủ thuật né từ khóa (chèn ký tự lạ) hoặc khi gặp các cụm từ không có trong từ điển gốc.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Model tụt nhiều nhất và ở nhóm case nào:
Model tụt nhiều nhất: Các mô hình truyền thống dựa trên đặc trưng Bag-of-Words (BoW + LogReg hoặc BoW + MLP). Tổng độ chính xác của chúng sụt giảm nghiêm trọng, chỉ còn khoảng 50% - 52% trên 50 case khó.

Nhóm case tụt nhiều nhất: Nhóm "Né từ khoá" (Obfuscation) (độ chính xác chạm đáy ở mức 0% - 17%) và nhóm "Spam giọng tự nhiên / Tin giao dịch hợp lệ".
2 Case cụ thể và cơ chế khiến model sai:

Case 1 (Nhóm Né từ khóa): "W-I-N a f.r.e.e g_i_f_t c_a_r_d today! C.l.i.c.k h.e.r.e" (Nhãn thật: Spam)

Cơ chế sai: Mô hình BoW hoạt động dựa trên việc khớp chính xác và đếm tần suất từ vựng (token). Khi kẻ gian cố tình chèn các dấu câu (-, ., _) vào giữa các chữ cái, các từ khóa quen thuộc biến thành những chuỗi ký tự lạ lẫm (f.r.e.e, w-i-n) hoàn toàn không tồn tại trong từ điển huấn luyện (vocabulary). Do bị "mù từ vựng", BoW không nhận diện được bất kỳ dấu hiệu rác nào và đoán nhầm đây là tin nhắn hợp lệ (Ham).

Case 2 (Nhóm Tin hợp lệ chứa từ nhạy cảm / Ham looks Spam): "Hey team, just a reminder about our marketing meeting to discuss the 'Win a Free iPhone' promotional campaign." (Nhãn thật: Ham)

Cơ chế sai: Mô hình BoW tách rời câu thành các từ đơn lẻ và bỏ qua hoàn toàn ngữ cảnh học (không hiểu được cụm "marketing meeting to discuss"). Mặc dù đây là một đoạn chat công việc hợp lệ, BoW lại lập tức "bắt" được các từ khóa tử huyệt mang trọng số spam cực cao như Win, Free, iPhone. Việc đếm các từ này cộng dồn điểm spam vượt qua ngưỡng quyết định, khiến mô hình tự động chặn nhầm một tin nhắn quan trọng (False Positive).

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1
Khi ngưỡng là 0.2: Mô hình sẽ dễ dàng gán nhãn là spam hơn. Điều này dẫn đến việc Recall tăng (ít bỏ sót spam hơn) nhưng Precision giảm mạnh (nhiều tin ham bị gán nhãn nhầm thành spam). F1 thường sẽ giảm do sự sụt giảm lớn của Precision.  
Khi ngưỡng là 0.8: Mô hình trở nên khắt khe hơn khi phân loại spam. Precision sẽ tăng (những tin bị gán nhãn spam có khả năng rất cao là spam thật), nhưng Recall giảm mạnh (nhiều tin spam bị bỏ sót).  
Với ngân hàng sợ chặn nhầm OTP: Nên chọn ngưỡng cao. Việc chặn nhầm một tin nhắn quan trọng như mã OTP (False Positive) có thể gây hậu quả nghiêm trọng hơn rất nhiều so với việc để lọt một vài tin nhắn rác(False Negative).  
ML · Giảm vocab còn 100 từ và tăng lên 5000 từ.
Vocab 100 từ: Từ vựng quá ít, khiến mô hình gặp tình trạng underfitting (chưa học đủ đặc trưng). Điểm F1 test sẽ giảm mạnh do mô hình thiếu thông tin để phân biệt các lớp.
Vocab 5000 từ: Việc tăng vocab lên quá cao có thể giúp mô hình học thêm được nhiều từ khóa đặc thù. Tuy nhiên, nó cũng làm tăng nguy cơ overfitting (học quá mức vào nhiễu). Mặc dù điểm F1 có thể không giảm nhiều, nhưng thời gian huấn luyện và độ phức tạp tính toán sẽ tăng lên đáng kể mà không mang lại lợi ích tương ứng.  
DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron
Mạng 4 nơ-ron: Số tham số ít, huấn luyện nhanh, nhưng F1 test thấp do mô hình không đủ sức mạnh biểu diễn (underfitting).
Mạng 512 nơ-ron: Số tham số lớn, huấn luyện chậm và có nguy cơ overfitting cao. 
Khi nào nên chọn mạng nhỏ: Khi tài nguyên tính toán (như RAM, CPU) bị giới hạn  hoặc khi yêu cầu về độ trễ (latency) trong suy luận phải cực kỳ thấp (real-time).*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Lựa chọn mô hình: Embedding + Logistic Regression (hoặc MLP head).

Dẫn chứng số liệu:

Về độ trễ (Latency): Mô hình đạt tốc độ suy luận cực nhanh, hoàn toàn đáp ứng SLA p95 < 50 ms/tin vì việc nhân ma trận diễn ra cục bộ (local). Trong khi đó, mọi phương án dùng LLM (zero-shot, few-shot) đều thất bại tiêu chí này (p95 thường > 500ms).

Về chất lượng: F1 trên tập test thường đạt ~0.947 và trên tập 50 case khó đạt ~0.640 - 0.731, vượt trội hơn hẳn so với BoW truyền thống (chỉ đạt ~0.33 trên case khó).

Về chi phí: Chạy cục bộ tốn $0, trong khi LLM tốn hàng nghìn USD/tháng (~$3,150/tháng cho 10 triệu tin/ngày).

Đánh đổi (Trade-off): Chấp nhận đánh đổi tỷ lệ nhận diện sai ở những ca "thao túng tâm lý" hoặc "prompt injection" vô cùng tinh vi (mà chỉ LLM mới đạt 100% accuracy) để đổi lấy sự tối ưu tuyệt đối về tốc độ thời gian thực và chi phí vận hành.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
AI Log:

Hỏi: Lỗi hệ thống treo 50 phút không chạy xong phần 4a/4b, và lỗi 429 RateLimitError của Gemini Free Tier.
Điều hiểu ra hôm nay:

Điểm F1 cao trên tập test thông thường mang tính "ảo tưởng" rất lớn vì nó cùng phân phối với tập huấn luyện. Sự bền bỉ (robustness) thực sự của hệ thống chỉ bộc lộ khi đối mặt với dữ liệu đối kháng (stress-test).

Dù LLM là "vua" về đọc hiểu ngữ nghĩa (đạt 100% case khó), nhưng trong kỹ thuật hệ thống (Engineering), nó bị giới hạn vật lý bởi độ trễ mạng và giới hạn kinh tế bởi chi phí token, khiến nó không thể làm màng lọc tuyến đầu cho hàng chục triệu tin nhắn mỗi ngày.
