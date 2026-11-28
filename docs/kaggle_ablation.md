# Chạy ablation multi-task trên nhiều notebook Kaggle

Ablation so sánh mô hình multi-task gốc (`base`) với 5 cải tiến, mỗi biến thể bật một cải tiến, cộng một
biến thể bật cả 5 (`all`), trên encoder RoBERTa-base, mỗi biến thể chạy 3 seed:

| Biến thể | Cải tiến | Cờ |
|---|---|---|
| `base` | Mô hình gốc | `--soft` |
| `hier` | (8) Aspect hai tầng | `--aspect-head hier` |
| `sent` | (10) Vector câu cho aspect | `--sentence-context` |
| `conf` | (11) Trọng số theo độ tin cậy LLM | `--aspect-confidence` |
| `ord` | (5) Sentiment có thứ tự | `--sentiment-loss ordinal` |
| `mark` | (12) Levitated markers | `--markers` |
| `all` | Cả 5 | tất cả các cờ trên |

Tổng cộng 7 biến thể × 3 seed = 21 run, mỗi run khoảng 16–19 phút trên T4 (10 epoch), tức khoảng 6 giờ GPU.
Chia ra nhiều notebook chạy song song giúp xong sớm hơn, nhưng **không** tiết kiệm quota GPU: tổng vẫn khoảng 6 giờ
trên quota 30 giờ mỗi tuần của Kaggle.

## 1. Chọn cách chia

Mỗi run ghi vào thư mục riêng `runs/ablation/multitask/<biến thể>-s<seed>/`, nên chia theo cách nào cũng gộp lại được.
Chỉ cần **mỗi cặp (biến thể, seed) chạy đúng một lần**.

| Cách | Số notebook | Ô tham số của từng notebook | Thời gian mỗi notebook |
|---|---|---|---|
| **A. Theo seed** (khuyên dùng) | 3 | `SEEDS="42"` / `"43"` / `"44"`, `VARIANTS` để nguyên | ~2 giờ |
| B. Theo nhóm biến thể | 2 | `VARIANTS="base hier sent conf"` và `VARIANTS="ord mark all"`, `SEEDS` để nguyên | ~3,5 giờ và ~3 giờ |
| C. Tối đa song song | 7 | `VARIANTS="base"`, `"hier"`, …, `"all"`, `SEEDS` để nguyên | ~1 giờ |

Kaggle giới hạn số phiên GPU chạy song song trên mỗi tài khoản. Nếu notebook mới không khởi động được, chờ một notebook
khác chạy xong rồi bấm chạy lại.

Nếu lượt seed (`scripts/run_seeds.sh`) cho thấy 20 epoch tốt hơn 10 epoch, đặt `EPOCHS = "20"` ở **tất cả** notebook.
Khi đó thời gian mỗi run gần gấp đôi. Mọi notebook phải dùng cùng `EPOCHS` và `MODEL`, nếu không kết quả không so được.

## 2. Tạo notebook (làm lần lượt cho từng notebook)

1. Vào Kaggle, chọn **Create → New Notebook**.
2. Chọn **File → Import Notebook → GitHub**, dán đường dẫn:
   `https://github.com/ptrnghieu/SEntFiN/blob/claude/adoring-fermat-r9ta0k/notebooks/kaggle_ablation.ipynb`
   (hoặc tải file `notebooks/kaggle_ablation.ipynb` về máy rồi chọn **File → Import Notebook → Upload**).
3. Đặt tên dễ nhận ra, ví dụ `ablation-s42`, `ablation-s43`, `ablation-s44` (bấm vào tên ở góc trên bên trái).
4. Mở panel bên phải:
   - **Settings → Accelerator: GPU T4 x2.**
   - **Settings → Internet: On.** Có thể Kaggle yêu cầu xác minh số điện thoại.
   - **Add-ons → Secrets:** tick secret **`HF_TOKEN`** cho notebook này. Mỗi notebook mới phải tick lại; secret
     không tự gắn theo khi tạo hoặc copy notebook.
5. **Chỉ sửa ô code đầu tiên** theo cách chia đã chọn. Ví dụ cách A, notebook thứ hai:
   ```python
   os.environ["SEEDS"] = "43"
   os.environ["VARIANTS"] = "base hier sent conf ord mark all"
   os.environ["EPOCHS"] = "10"
   os.environ["MODEL"] = "roberta-base"
   ```
6. Bấm **Save Version → Save & Run All (Commit) → Save.** Phiên bản commit vẫn chạy khi đóng tab, giới hạn 12 giờ.

Không cần thêm dataset nào qua "Add Input": notebook tự tải code từ GitHub và dữ liệu, mô hình từ Hugging Face.

## 3. Theo dõi

- Trang notebook → **Versions**: trạng thái `Running`, rồi `Complete` hoặc `Error`.
- Ô chạy chính dùng `%%bash`, nên output thường chỉ hiện khi ô đó chạy xong. Không thấy log mới trong vài giờ là bình thường.
- Nếu chạy lỗi sớm (trong vài phút đầu), xem log ở Versions → Logs. Nguyên nhân hay gặp:
  - `HF_TOKEN` chưa được tick cho notebook: lỗi ở ô đọc secret.
  - Chưa bật Internet: lỗi ở `git clone` hoặc `pip install`.
  - Chưa bật GPU: dừng ở dòng `assert torch.cuda.is_available()`.

## 4. Khi một notebook bị dừng giữa chừng

Mỗi run hoàn tất đều đã ghi `metrics.json`, và `SEEDS.md` được cập nhật sau mỗi seed. Chỉ cần chạy lại phần còn thiếu:
xem danh sách đã xong ở ô cuối (`find runs/ablation -name metrics.json`), rồi tạo một notebook mới với `VARIANTS`
và `SEEDS` chỉ gồm các cặp còn thiếu. Ví dụ notebook seed 43 dừng sau `conf`:
```python
os.environ["SEEDS"] = "43"
os.environ["VARIANTS"] = "ord mark all"
```

## 5. Gom kết quả

**Cách 1, nhờ Claude:** báo tên các notebook đã chạy xong. Claude tải `runs/ablation/` của từng notebook qua Kaggle
API về chung một thư mục, rồi chạy `python -m absa_models.aggregate` để ra bảng trung bình ± độ lệch chuẩn và
phân tích.

**Cách 2, tự làm:**
1. Ở mỗi notebook: tab **Output**, tải thư mục `SEntFiN/runs/ablation/multitask/`.
2. Giải nén tất cả vào **cùng** một thư mục `runs/ablation/multitask/` trong repo (các thư mục con không trùng tên).
3. Chạy:
   ```bash
   python -m absa_models.aggregate runs/ablation
   ```
   Kết quả ở `runs/ablation/SEEDS.md`.

Mỗi notebook cũng có `SEEDS.md` riêng, nhưng chỉ chứa phần việc của notebook đó. Bảng cuối cùng phải gom từ tất cả.

## 6. Đọc kết quả

- Cột chính: `all triplet · f1` (F1 bộ ba trên toàn bộ tập test), dạng trung bình ± độ lệch chuẩn qua 3 seed.
- Một cải tiến chỉ đáng tin khi mức tăng so với `base` lớn hơn rõ độ lệch chuẩn. Các lần chạy lại RoBERTa với cùng
  cấu hình đã lệch nhau tới khoảng 2 điểm.
- Xem thêm các cột theo mục tiêu của từng cải tiến: `span+aspect` cho `hier`/`sent`/`conf`, `span+sentiment` cho `ord`,
  `finentity triplet` cho `mark` (nơi pipeline đang hơn multi-task).
