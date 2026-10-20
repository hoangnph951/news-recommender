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
MIND small (Microsoft News): khoảng 50.000 user, 65.000 tin, 230.000 lần hiển thị có nhãn
click / không click.

## 4. Phương pháp dự kiến
1. Baseline: tin phổ biến; gợi ý dựa trên nội dung (TF-IDF + cosine); lai có trọng số.
2. Mô hình chính: NRMS, gồm news encoder và user encoder dùng multi-head self-attention,
   cùng click predictor là tích vô hướng. Huấn luyện bằng negative sampling (1 positive, K negative).
3. Mở rộng (nếu kịp): LSTUR (sở thích ngắn hạn và dài hạn), học tương phản cho user encoder.

## 5. Đánh giá
AUC, MRR, nDCG@5, nDCG@10 trên tập dev của MIND; so sánh giữa baseline và NRMS.

## 6. Sản phẩm
Mã nguồn, demo web (chọn user, xem gợi ý, bấm đọc tin), báo cáo PDF và slide thuyết trình.
