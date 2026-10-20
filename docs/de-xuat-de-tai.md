# Đề xuất đề tài: Hệ gợi ý tin tức cá nhân hoá

> Bản nháp theo yêu cầu slide L0 (1–2 trang). Điền thông tin nhóm rồi xuất ra PDF.

**Nhóm:** Họ tên – MSSV – Email (2–4 thành viên)

## 1. Bài toán thực tế
Mỗi ngày các trang báo điện tử đăng rất nhiều tin, nên người đọc khó tìm được tin mình quan tâm
(quá tải thông tin). Mục tiêu là xây dựng hệ thống gợi ý các bản tin phù hợp với từng người
dựa trên lịch sử đọc của họ.

**Kịch bản ứng dụng:** người dùng mở trang chủ và thấy mục "Gợi ý cho bạn" gồm 10 tin. Mỗi khi
họ đọc một tin, danh sách gợi ý được cập nhật lại.

## 2. Input / Output
- **Input:** lịch sử click của user (danh sách tin đã đọc theo thời gian) và nội dung tin
  (tiêu đề, tóm tắt, chuyên mục, chuyên mục con).
- **Output:** danh sách top-K tin chưa đọc, xếp theo điểm dự đoán khả năng click.

## 3. Dữ liệu
MIND-large (Microsoft News): khoảng 1 triệu user, 161.000 tin, 2,2 triệu impression train và
376.000 impression dev có nhãn click / không click; kèm entity Wikidata của từng tin.

## 4. Phương pháp dự kiến
1. Baseline: tin phổ biến, phổ biến 24h gần nhất, gợi ý dựa trên nội dung (TF-IDF); tái lập NRMS và NAML (GloVe).
2. Mô hình đề xuất: news encoder dùng mô hình ngôn ngữ pretrain (MiniLM) trên tiêu đề và tóm tắt,
   kết hợp các view chuyên mục và entity; user encoder self-attention có thêm loss học tương phản;
   click predictor kết hợp độ phù hợp nội dung với độ phổ biến / độ mới của tin theo thời gian.
3. Ablation từng thành phần, 3 seed và paired bootstrap để chứng minh mức cải thiện.

## 5. Đánh giá
AUC, MRR, nDCG@5, nDCG@10 trên tập dev MIND-large, xếp hạng toàn bộ ứng viên của mỗi impression; so sánh với NRMS tái lập và tham chiếu kết quả trong slide L9 (NRMS: AUC 67.76).

## 6. Sản phẩm
Mã nguồn, demo web (gợi ý kèm lý do, phản hồi click, trang kết quả thực nghiệm), báo cáo PDF và slide thuyết trình.
