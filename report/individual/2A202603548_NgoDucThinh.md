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

Bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin)Bảng 1: Các chỉ số phân loại (Chất lượng mô hình)Mô hìnhChỉ số 1Chỉ số 2Chỉ số 3Chỉ số 4 (F1)1. BoW + LogReg (tự code)0.9890.7840.8740.8892. BoW + MLP (deep learning)0.9800.8740.9240.9193a. Encoder (MiniLM) + kNN0.9040.9280.9160.9473b. Encoder (MiniLM) + LogReg0.9810.9190.9490.9473c. Encoder (MiniLM) + MLP head0.9810.9190.9490.9474a. LLM nhỏ zero-shot1.0000.9500.9740.9744b. LLM lớn zero-shot1.0000.9500.9740.9744c. LLM nhỏ few-shot1.0000.9000.9470.947   

Bảng 2: Các chỉ số hiệu năng (Tốc độ & Chi phí)Mô hìnhTrain (s)Tin/s (cả lô)p50 ms/tinp95 ms/tinUSD / 1k tin1. BoW + LogReg4.39418,098.120.0570.108$0.0002. BoW + MLP10.60026,020.050.3052.555$0.0003a. Encoder + kNN100.14129.1551.69170.433$0.0003b. Encoder + LogReg100.61446.0329.53446.631$0.0003c. Encoder + MLP head101.65344.3428.07136.779$0.0004a. LLM nhỏ zero-shot0.0001.53565.8771239.503$0.0194b. LLM lớn zero-shot0.0001.45526.0761424.687$0.0444c. LLM nhỏ few-shot0.0001.34516.8592097.892$0.058
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

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
