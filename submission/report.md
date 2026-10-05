Tôi sử dụng AWS, region us-east-1, Compute Node t3.medium (2 vCPU, 4 GB RAM), source commit 40f507acbbfba207e6cee0e77a31604bae74a3a5.

Dataset Credit Card Fraud Detection có 284,807 dòng. Dữ liệu được tách 80% train+validation / 20% test với stratify=y; phần train+validation tiếp tục được tách validation với tỷ lệ 0.2. Seed là 42.

Load dữ liệu mất 1.891 giây; training mất 2.807 giây; best iteration là 1.

Trên tập test, AUC = 0.9391, Accuracy = 0.9992, F1 = 0.7692, Precision = 0.7732, Recall = 0.7653.

Latency dự đoán 1 dòng là 0.807 ms, đo sau warm-up và lấy trung bình 1,000 lần. Throughput batch 1,000 dòng đạt khoảng 965,260 dòng/giây, tính bằng số dòng chia cho thời gian inference của batch.

Khi kiểm tra tài nguyên sau benchmark, CPU gần như idle (100% idle, load average 0.01/0.07/0.04). RAM sử dụng khoảng 231 MiB / 3.7 GiB, còn 3.3 GiB available, không sử dụng swap.

Billing đã có bằng chứng trong screenshots/billing.png

Tôi đã lưu kết quả benchmark và bằng chứng cần thiết, sau đó chạy terraform destroy lúc 10:31 ngày 04/10/2026 để dọn dẹp tài nguyên AWS. 
