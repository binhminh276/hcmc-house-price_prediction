# Dự Đoán Giá Bất Động Sản Tại TP.HCM

## Giới thiệu dự án
Đây là đồ án môn học tập trung vào việc ứng dụng các kỹ thuật khai phá dữ liệu (Data Mining) và học máy (Machine Learning) để phân tích và dự đoán giá nhà đất trên địa bàn Thành phố Hồ Chí Minh.

Mục tiêu của dự án là xây dựng một quy trình hoàn chỉnh từ khâu thu thập dữ liệu thô trên nền tảng web, tiền xử lý, phân tích khám phá (EDA), cho đến trực quan hóa dữ liệu qua dashboard và triển khai mô hình học máy thực tế.

## Nguồn tài nguyên và Demo triển khai
Dataset, Model và Demo giao diện:
* **Hugging Face Space:** [demo-price-prediction](https://huggingface.co/spaces/binhminh276/demo-price-prediction)

## Quy trình thực hiện
Dự án được chia thành các giai đoạn chính sau:

1. **Thu thập dữ liệu (Web Scraping):** * Xây dựng script để cào dữ liệu tự động từ trang web [batdongsan.com.vn](https://batdongsan.com.vn/).
   * Thu thập các đặc trưng quan trọng của bất động sản như diện tích, vị trí, số phòng, và giá bán...

2. **Tiền xử lý dữ liệu (Data Cleaning):**
   * Xử lý các giá trị bị thiếu (missing values) và nhiễu.
   * Chuẩn hóa các định dạng văn bản và chuyển đổi kiểu dữ liệu phù hợp cho việc phân tích.

3. **Phân tích khám phá dữ liệu (EDA - Exploratory Data Analysis):**
   * Khám phá sự phân bố của giá bất động sản theo từng khu vực tại TP.HCM.
   * Tìm ra mối tương quan giữa các đặc trưng đầu vào và giá nhà để chọn lọc các biến quan trọng nhất.

4. **Trực quan hóa dữ liệu (Dashboard):**
   * Xây dựng báo cáo trực quan (dashboard) để cung cấp cái nhìn tổng quan và sâu sắc nhất về thị trường bất động sản dựa trên tập dữ liệu đã thu thập.

5. **Xây dựng và triển khai mô hình (Modeling & Deployment):**
   * Áp dụng các thuật toán học máy (bao gồm cả phân cụm K-means như trong dữ liệu đã xử lý) để dự đoán mức giá.
   * Triển khai ứng dụng hoàn chỉnh để người dùng cuối có thể tương tác.

## Thành viên nhóm
* **Nguyễn Thanh Bình Minh** - MSSV: 23110266
* **Trần Quang Duy** - MSSV: 23110195
* **Nguyễn Phúc An** - MSSV: 23110175
