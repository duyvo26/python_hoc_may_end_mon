# Giả lập I/O cho vòng lặp phân tích K tốt nhất

File này giả lập các giá trị Input/Output (I/O) chạy qua đoạn code từ dòng 44 đến 81 để minh họa quá trình hệ thống tự động tìm K.

Giả sử `K_range = range(2, 6)` (tức là thử các K = 2, 3, 4, 5) để ví dụ ngắn gọn hơn. 
Trọng số cấu hình đang là:
- `w_sil = 2`
- `w_db = 2` 
- `w_ch = 1`
- `w_elbow = 1`

---

## 1. Vòng lặp `for k in K_range:`
Code sẽ tính toán từng giá trị K và đưa kết quả vào mảng.

**Khi `k = 2`:**
- `km.inertia_` = 1500 => `wcss` = [1500]
- `silhouette_score` = 0.45 => `sil_km` = [0.45]
- `davies_bouldin` = 0.80 => `db_km` = [0.80]
- `calinski_harabasz` = 120 => `ch_km` = [120]

**Khi `k = 3`:**
- `km.inertia_` = 800 => `wcss` = [1500, 800]
- `silhouette_score` = 0.65 => `sil_km` = [0.45, 0.65]  *(điểm cao, có vẻ tốt)*
- `davies_bouldin` = 0.45 => `db_km` = [0.80, 0.45]  *(điểm thấp, có vẻ tốt)*
- `calinski_harabasz` = 190 => `ch_km` = [120, 190]

**Khi `k = 4`:**
- `km.inertia_` = 600 => `wcss` = [1500, 800, 600]
- `silhouette_score` = 0.55 => `sil_km` = [0.45, 0.65, 0.55]
- `davies_bouldin` = 0.60 => `db_km` = [0.80, 0.45, 0.60]
- `calinski_harabasz` = 250 => `ch_km` = [120, 190, 250] *(điểm cao nhất)*

**Khi `k = 5`:**
- `km.inertia_` = 550 => `wcss` = [1500, 800, 600, 550]
- `silhouette_score` = 0.50 => `sil_km` = [0.45, 0.65, 0.55, 0.50]
- `davies_bouldin` = 0.70 => `db_km` = [0.80, 0.45, 0.60, 0.70]
- `calinski_harabasz` = 210 => `ch_km` = [120, 190, 250, 210]

---

## 2. Tìm giá trị Max / Min để chốt K cho từng chỉ số
Mảng `K_range` là: `[2, 3, 4, 5]` tương ứng với index `[0, 1, 2, 3]`.

- **Elbow:** Hàm `_detect_elbow_kneedle(wcss)` nhìn vào độ dốc của mảng `[1500, 800, 600, 550]` và thấy từ 1500 rớt xuống 800 là dốc nhất, tạo thành khuỷu tay tại K=3.
  => `idx_elbow = 1` -> `b_elbow = K_range[1] = 3`

- **Silhouette (Tìm Max):** Mảng là `[0.45, 0.65, 0.55, 0.50]`
  => Max là 0.65 ở index 1.
  => `np.argmax(sil_km)` = 1 -> `b_sil_km = K_range[1] = 3`

- **Davies-Bouldin (Tìm Min):** Mảng là `[0.80, 0.45, 0.60, 0.70]`
  => Min là 0.45 ở index 1.
  => `np.argmin(db_km)` = 1 -> `b_db_km = K_range[1] = 3`

- **Calinski-Harabasz (Tìm Max):** Mảng là `[120, 190, 250, 210]`
  => Max là 250 ở index 2.
  => `np.argmax(ch_km)` = 2 -> `b_ch_km = K_range[2] = 4`

---

## 3. Biểu quyết có trọng số (Weighted Voting)

Dựa theo kết quả tính được, các K tốt nhất (best K) là:
- Silhouette vote: 3
- Davies-Bouldin vote: 3
- Calinski-Harabasz vote: 4
- Elbow vote: 3

Hệ thống tính `km_votes` như sau:
```python
km_votes = [3]*2 + [3]*2 + [4]*1 + [3]*1
# Hay diễn giải ra:
km_votes = [3, 3] + [3, 3] + [4] + [3]
km_votes = [3, 3, 3, 3, 4, 3]
```

## 4. Chốt kết quả vòng lặp
Sử dụng `Counter`:
```python
Counter([3, 3, 3, 3, 4, 3]) 
# Trả về: {3: 5, 4: 1}  -> Có 5 phiếu cho K=3 và 1 phiếu cho K=4.
```

Hàm `most_common(1)[0][0]` sẽ lấy phần tử có số lượng lớn nhất, chính là số `3`.
=> **`km_trial_k` = 3**

Thuật toán lưu lại K=3 cho lần thử nghiệm này (trial `t`). Nếu có nhiều lượt thử (`n_trials > 1`), thuật toán sẽ lặp lại toàn bộ quá trình này và cuối cùng lại đem các K thắng ở từng lượt ra biểu quyết vòng chung kết.
