# Email Spam Detection

Dự án phân loại email Spam/Ham sử dụng mô hình Machine Learning kết hợp TF-IDF.  
Ứng dụng được xây dựng phục vụ môn Phân Tích Dữ Liệu và có thể triển khai như một công cụ thực tế.

---

## Giới thiệu

Hệ thống được thiết kế để dự đoán xem một email có phải là spam hay không dựa trên nội dung văn bản.  
Phương pháp tiếp cận bao gồm:

- Tiền xử lý văn bản
- Vector hóa bằng TF-IDF
- Huấn luyện mô hình Logistic Regression
- Xây dựng giao diện bằng Streamlit

---

## Tính năng

### 1. Phân loại email theo thời gian thực
- Nhập nội dung email bất kỳ để kiểm tra.
- Kết quả trả về dạng Spam hoặc Ham.
- Hiển thị độ tự tin của mô hình.

### 2. Phân loại hàng loạt từ file CSV
- Cho phép tải lên file chứa nhiều email.
- Tự động phân loại toàn bộ.
- Xuất kết quả ra file CSV.
- Có bảng biểu đồ đánh giá mô hình (confusion matrix, phân bố nhãn).

### 3. Dashboard tổng quan
- Giới thiệu về mô hình, dữ liệu, phương pháp.
- Ví dụ trực quan về phân bố Spam/Ham.

---

## Công nghệ sử dụng

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib, Seaborn
- Joblib (lưu mô hình)

---

## Cách chạy dự án

### Bước 1: Cài đặt thư viện
