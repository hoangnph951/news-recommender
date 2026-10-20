# Hệ gợi ý tin tức (News Recommender)

Đồ án môn **Hệ gợi ý (IT4613)**. Hệ thống gợi ý bản tin cá nhân hoá, huấn luyện và đánh giá trên
**MIND-large** của Microsoft News: 161k tin, 2,2 triệu impression train, 376k impression dev.

- **Input**: lịch sử click của user và nội dung tin (tiêu đề, tóm tắt, chuyên mục, entity Wikidata).
- **Output**: danh sách tin xếp hạng theo khả năng click, kèm lý do gợi ý ("Vì bạn đã đọc…").

## Mô hình đề xuất

Giữ khung 3 thành phần của các mô hình trong slide L9 (news encoder, user encoder, click predictor)
và cải tiến từng thành phần:

```
title + abstract ─► MiniLM (cache 10 tầng dưới, fine-tune 2 tầng trên) ─► attention ─┐
category, subcategory ─► embedding ────────────────────────────────────────────────────┼─► attention giữa các view ─► vector tin
entity Wikidata ─► TransE embedding (entity_embedding.vec) ────────────────────────────┘
lịch sử 50 tin ─► multi-head self-attention + additive attention ─► vector user   (+ loss học tương phản)
điểm = (1 − η) · <user, tin> + η · popularity(tin, thời điểm)     η do user quyết định (user mới → η lớn)
```

| Cải tiến | So với | Slide |
|---|---|---|
| News encoder bằng mô hình ngôn ngữ pretrain (PLM) thay cho GloVe | NRMS, NAML, LSTUR | L9 |
| Nhiều view: văn bản, chuyên mục, chuyên mục con, entity | NAML, DKN | L9 |
| Độ phổ biến và độ mới theo thời gian thực, chỉ dùng dữ liệu **trước** thời điểm impression | PP-Rec, hybrid có trọng số | L7 |
| Học tương phản trên lịch sử đọc (mask, crop, reorder) | CL4Rec | L10 |
| Ensemble nhiều seed | | L7 |

Các lựa chọn khác được giữ để chạy ablation: user encoder `mean` (trung bình) / `cand` (candidate-aware),
loss `listwise` (cả impression) thay cho `sampled` (1 click + 4 tin không click).

## Giao thức đánh giá

- Độ đo **AUC, MRR, nDCG@5, nDCG@10** (L8), tính trên **toàn bộ ứng viên của mỗi impression gốc** giống
  script chính thức của MIND, rồi lấy trung bình theo impression. Báo cáo thêm hai nhóm: user đã có trong
  train (*overlap*) và user mới (*unseen*), như bảng của L9.
- Đánh giá trên **tập dev của MIND-large**. Tập test không công bố nhãn; bảng ở slide L9 là kết quả nộp
  leaderboard trên tập test, nên chỉ dùng để tham chiếu. So sánh chính là với **NRMS tự chạy lại** trong
  cùng điều kiện (`configs/nrms_glove.yaml`).
- Mô hình chính chạy **3 seed** (báo mean ± std) và có **paired bootstrap** theo impression để kiểm tra
  mức cải thiện có ý nghĩa thống kê không.
- Đặc trưng popularity chỉ đếm click và hiển thị ở các giờ **trước** giờ của impression, nên không dùng
  thông tin tương lai. Test `test_popularity_has_no_future_leakage` kiểm tra điều này.

Mốc cần vượt (slide L9, MIND-large test): NRMS AUC 67.76, MRR 33.05, nDCG@5 35.94, nDCG@10 41.63.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # máy có GPU: chọn bản CUDA
pip install -r requirements.txt
```

## Chạy

**Dữ liệu** (MIND trên HuggingFace là gated): đồng ý điều khoản ở
https://huggingface.co/datasets/yjw1029/MIND, tạo token Read rồi đăng nhập:

```bash
hf auth login
python scripts/download_mind.py --size large             # thêm --glove để có GloVe cho baseline NRMS
python -m newsrec.preprocess --size large                # thêm --glove data/raw/glove/glove.840B.300d.txt
```

**Huấn luyện** (cần GPU, nên dùng [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) trên Kaggle):

```bash
python -m newsrec.baselines                              # popularity, popularity 24h, TF-IDF (CPU)
python -m newsrec.plm_cache --device cuda                # cache PLM một lần
python -m newsrec.experiments main --seeds 42 43 44      # NRMS tái lập + mô hình đề xuất
python -m newsrec.experiments ablation --set train.epochs=1 train.max_train_samples=1000000
python -m newsrec.experiments --list                     # xem toàn bộ thực nghiệm
```

Mỗi run nằm trong `runs/<tên>_seed<seed>/`: `config.yaml`, `best.pt`, `metrics.json`, `dev_scores.npy`,
`news_vecs.npy`. Huấn luyện tự lưu `last.pt` khi chạm `train.time_limit_hours`; chạy lại cùng lệnh để tiếp tục.

**Phân tích:**

```bash
python -m newsrec.analysis table                         # bảng mean ± std -> results.md
python -m newsrec.analysis ensemble runs/full_seed42 runs/full_seed43 runs/full_seed44 --name ens_full
python -m newsrec.analysis bootstrap runs/full_seed42 runs/nrms_glove_seed42
```

**Demo web** (sau khi giải nén `results_package.zip` từ Kaggle vào thư mục repo):

```bash
uvicorn app.main:app
```

Mở http://127.0.0.1:8000: chọn user (hoặc "user mới" để xem cold-start), xem gợi ý kèm lý do và trọng số
độ phổ biến, bấm đọc tin để gợi ý cập nhật. Click được lưu vào `data/feedback.db`. Trang `/results`
hiển thị bảng kết quả của mọi run. Đổi mô hình bằng biến môi trường `NEWSREC_RUN=runs/<tên run>`.

## Test

```bash
python -m pytest
```

Test chạy trên bộ dữ liệu giả lập cùng định dạng MIND và một BERT tí hon tạo ngay trên máy, nên không cần
tải gì. Phạm vi test: tiền xử lý, không rò rỉ thời gian, các biến thể mô hình, huấn luyện có resume, baseline,
ensemble, bootstrap và demo web.

## Cấu trúc

```
newsrec/
  mind.py         đọc news.tsv, behaviors.tsv, entity_embedding.vec
  preprocess.py   chuyển MIND thành mảng numpy phẳng (data/processed/mind_large)
  data.py         nạp dữ liệu, dataset sampled / listwise
  plm_cache.py    cache hidden state của PLM
  model.py        news encoder, user encoder, popularity, mô hình đầy đủ
  losses.py       ranking loss, học tương phản
  train.py        huấn luyện (AMP, warmup, resume, chọn checkpoint theo AUC dev)
  evaluate.py     chấm điểm toàn bộ impression, độ đo theo nhóm user, file nộp leaderboard
  metrics.py      AUC, MRR, nDCG
  baselines.py    popularity, popularity 24h, TF-IDF
  experiments.py  danh sách thực nghiệm (main, ablation) và trình chạy
  analysis.py     bảng kết quả, ensemble, paired bootstrap
configs/          base.yaml (mô hình đề xuất), nrms_glove.yaml, naml_glove.yaml
app/              demo FastAPI + giao diện web + trang kết quả
notebooks/        kaggle_train.ipynb
scripts/          tải dữ liệu
tests/
docs/             đề xuất đề tài
```

## Tài liệu tham khảo

- Wu et al. 2019. *Neural News Recommendation with Multi-Head Self-Attention* (NRMS). EMNLP.
- Wu et al. 2019. *Neural News Recommendation with Attentive Multi-View Learning* (NAML). IJCAI.
- Wu et al. 2020. *MIND: A Large-scale Dataset for News Recommendation*. ACL.
- Wu et al. 2021. *Empowering News Recommendation with Pre-trained Language Models* (PLM-NR). SIGIR.
- Qi et al. 2021. *PP-Rec: News Recommendation with Personalized User Interest and Time-aware News Popularity*. ACL.
- Xie et al. 2022. *Contrastive Learning for Sequential Recommendation* (CL4Rec). ICDE.
- Iana et al. 2023. *Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives*. SIGIR.
- Wang et al. 2020. *MiniLM*. NeurIPS. Mô hình dùng: `sentence-transformers/all-MiniLM-L12-v2`.
