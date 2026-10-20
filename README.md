# Hệ gợi ý tin tức (News Recommender)

Đồ án môn **Hệ gợi ý (IT4613)**. Hệ thống gợi ý bản tin cho người đọc dựa trên lịch sử click,
dùng bộ dữ liệu [MIND](https://msnews.github.io/) của Microsoft News.

- **Input**: lịch sử các tin user đã click, cùng nội dung tin (tiêu đề, tóm tắt, chuyên mục).
- **Output**: danh sách top-K tin được xếp hạng theo khả năng user sẽ click.

## Phương pháp và slide tương ứng

| Thành phần | File | Kiến thức (slide) |
|---|---|---|
| Đọc và tiền xử lý MIND, negative sampling | `newsrec/mind.py`, `newsrec/preprocess.py`, `newsrec/dataset.py` | L9 |
| Baseline phổ biến (popularity) | `newsrec/baselines.py` | L1, L2 |
| Baseline dựa trên nội dung: TF-IDF + cosine | `newsrec/baselines.py` | L4 |
| Lai ghép có trọng số: TF-IDF + popularity | `newsrec/baselines.py` | L7 |
| **NRMS**: news encoder, user encoder, click predictor | `newsrec/nrms.py`, `newsrec/train.py` | L9 |
| Độ đo AUC, MRR, nDCG@5, nDCG@10 | `newsrec/metrics.py` | L8 |
| Demo web | `app/` | |

## Cài đặt

Cần Python 3.10 trở lên.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # hoặc bản CUDA nếu có GPU
pip install -r requirements.txt
```

## Chạy

```bash
# 1. Tải MIND small vào data/ (nếu lỗi, tải tay từ https://msnews.github.io/)
python scripts/download_mind.py

# 2. Tiền xử lý. Thêm --glove path/to/glove.6B.300d.txt để dùng word embedding pretrain
python -m newsrec.preprocess

# 3. Đánh giá các baseline trên tập dev
python -m newsrec.baselines

# 4. Huấn luyện NRMS (lưu checkpoint tốt nhất vào checkpoints/nrms.pt)
python -m newsrec.train --epochs 3
python -m newsrec.train --epochs 1 --max-train 5000 --max-dev 2000   # chạy thử nhanh

# 5. Demo web tại http://127.0.0.1:8000
uvicorn app.main:app --reload
```

Nếu chưa có checkpoint NRMS, demo tự chuyển sang baseline lai TF-IDF + popularity.

## Test

```bash
python -m pytest
```

Test chạy trên một bộ dữ liệu giả lập rất nhỏ cùng định dạng với MIND (`tests/conftest.py`),
nên không cần tải dữ liệu thật.

## Cấu trúc

```
newsrec/            mã nguồn chính
  mind.py           đọc news.tsv, behaviors.tsv
  preprocess.py     tạo data/processed/mind_small.pkl
  dataset.py        dataset huấn luyện (1 positive + K negative)
  baselines.py      popularity, tfidf, hybrid
  nrms.py           mô hình NRMS
  train.py          huấn luyện và đánh giá NRMS
  metrics.py        AUC, MRR, nDCG
app/                demo FastAPI và giao diện web
scripts/            tải dữ liệu
tests/              test trên dữ liệu giả lập
docs/               đề xuất đề tài, ghi chú báo cáo
```

## Hướng phát triển

- [ ] LSTUR: sở thích ngắn hạn và dài hạn, dùng GRU (L6, L9)
- [ ] NAML: thêm abstract và category vào news encoder (L9)
- [ ] Học tương phản cho user encoder: crop, mask, reorder lịch sử (L10)
- [ ] Switching hybrid: user mới dùng popularity hoặc chuyên mục tự chọn (L5, L7)
- [ ] Thay word embedding bằng BERT

## Tài liệu tham khảo

- Wu et al. 2019. *Neural News Recommendation with Multi-Head Self-Attention* (NRMS). EMNLP.
- Wu et al. 2020. *MIND: A Large-scale Dataset for News Recommendation*. ACL.
