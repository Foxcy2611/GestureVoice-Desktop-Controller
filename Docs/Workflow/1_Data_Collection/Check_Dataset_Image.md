# ✋ Kiểm tra tập dữ liệu train của model cử chỉ tay

Sau khi thu đủ 3000 mẫu cho 10 case, run script `Check_Dataset_Image.py` để kiểm tra tập dữ liệu

Một tập dữ liệu `.csv` đạt tiêu chuẩn khi đạt đủ các tiêu chuẩn sau

## 📊 Số lượng mẫu / nhãn

- Mỗi nhãn >= 95% x 300 (tức >= 285 mẫu)
- Khi thu mẫu, trong script có báo hiệu số lượng mẫu, nên cái này có thể đạt ngay từ đầu

## 🚫 NaN / Lỗi

- Không có giá trị NaN nào

## 🔍 Gần trùng lặp

- Mỗi nhãn < 5% cặp mẫu có khoảng cách Euclid < 0.02
- Do quá trình thu mẫu có thể tay không di chuyển quá xa mà đã thu mẫu

## 🤖 KNN baseline

- KNN kiểm tra `sanity check` thuần túy chất lượng data, vì KNN không có gì để train/tinh chỉnh
- Nếu KNN mà phân loại tệ thì data có vấn đề  
> Yêu cầu: Accuracy >= 85%

## 🧠 MLP baseline

- Là 1 mạng neuron dùng để training model
- Sử dụng MLP nhưng cơ bản nhỏ  
> Yêu cầu Accuracy >= 85%

# 📚 Tham khảo
 

``` text
============================================================
1. SỐ LƯỢNG MẪU THEO NHÃN
============================================================
   left_one       :  300  [ĐỦ]
   left_two       :  300  [ĐỦ]
   right_fist     :  300  [ĐỦ]
   right_five     :  300  [ĐỦ]
   right_four     :  300  [ĐỦ]
   right_like     :  300  [ĐỦ]
   right_ok       :  300  [ĐỦ]
   right_one      :  300  [ĐỦ]
   right_three    :  300  [ĐỦ]
   right_two      :  300  [ĐỦ]
   Tổng số nhãn: 10 (kỳ vọng 10)
   Tổng số mẫu : 3000 (kỳ vọng 3000)

============================================================
2. KIỂM TRA NaN / GIÁ TRỊ LỖI
============================================================
   Không có giá trị NaN. OK.

============================================================
3. KIỂM TRA GẦN TRÙNG LẶP (ngưỡng khoảng cách < 0.02)
============================================================
   left_one       :   0.0% cặp gần trùng lặp  [OK]
   left_two       :   0.0% cặp gần trùng lặp  [OK]
   right_fist     :   0.0% cặp gần trùng lặp  [OK]
   right_five     :   0.0% cặp gần trùng lặp  [OK]
   right_four     :   0.0% cặp gần trùng lặp  [OK]
   right_like     :   0.0% cặp gần trùng lặp  [OK]
   right_ok       :   0.0% cặp gần trùng lặp  [OK]
   right_one      :   0.0% cặp gần trùng lặp  [OK]
   right_three    :   0.0% cặp gần trùng lặp  [OK]
   right_two      :   0.0% cặp gần trùng lặp  [OK]

============================================================
4. KIỂM TRA THỬ - KNN + MLP (chưa phải model chính thức)
============================================================
   KNN độ chính xác (test 20%): 97.33%
   MLP độ chính xác (test 20%): 99.00%
   OK - cả 2 baseline đều tốt, dữ liệu sẵn sàng để train model chính thức.

   Chi tiết MLP theo từng nhãn:
              precision    recall  f1-score   support

    left_one       1.00      1.00      1.00        60
    left_two       0.98      1.00      0.99        60
  right_fist       1.00      0.97      0.98        60
  right_five       0.98      0.97      0.97        60
  right_four       0.98      1.00      0.99        60
  right_like       0.97      1.00      0.98        60
    right_ok       0.98      0.98      0.98        60
   right_one       1.00      1.00      1.00        60
 right_three       1.00      1.00      1.00        60
   right_two       1.00      0.98      0.99        60

    accuracy                           0.99       600
   macro avg       0.99      0.99      0.99       600
weighted avg       0.99      0.99      0.99       600
```