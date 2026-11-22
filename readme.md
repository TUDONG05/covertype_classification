# Phân loại lớp phủ rừng quy mô lớn — Covertype

Đồ án học phần **Phân tích dữ liệu lớn · Nhóm 11**, nghiên cứu phân loại lớp phủ rừng bằng **Logistic Regression (LR)** và **Random Forest (RF)** trên 581.012 quan sát. Dự án triển khai hai thực nghiệm: sử dụng toàn bộ 54 đặc trưng và chọn 20 đặc trưng bằng Mutual Information.

**Kết quả nổi bật:** RF sau tối ưu với 54 đặc trưng đạt **Accuracy 96,81%**, **Macro-F1 0,9435**. Khi giảm còn 20 đặc trưng, RF đạt **Accuracy 96,37%**, **Macro-F1 0,9368**.

> Các số liệu dưới đây được tổng hợp từ output đã lưu trong hai notebook, không phải kết quả huấn luyện lại khi tạo repository. Điểm trong bảng làm tròn 4 chữ số thập phân; Accuracy trong bảng nằm trên thang 0–1.

## 1. Mục tiêu đề tài

- Dự đoán một trong bảy lớp phủ rừng từ thông tin địa hình, thủy văn, độ che sáng, khu vực hoang dã và loại đất.
- Khảo sát chất lượng dữ liệu và sự mất cân bằng giữa các lớp.
- So sánh mô hình tuyến tính LR với mô hình tổ hợp cây RF trên cùng quy trình đánh giá.
- Đánh giá tác động của GridSearchCV và chọn đặc trưng tới chất lượng dự đoán, thời gian huấn luyện và suy luận.
- Lưu pipeline, dự đoán, cấu hình và dấu vân tay dữ liệu để hỗ trợ kiểm tra lại thực nghiệm.

“Quy mô lớn” trong phạm vi dự án là xử lý hơn nửa triệu quan sát bằng Python/scikit-learn trên một máy, giới hạn tối đa hai luồng; dự án chưa triển khai Spark, Hadoop hoặc huấn luyện phân tán.

## 2. Dữ liệu Covertype

Notebook tải dữ liệu từ **UCI Machine Learning Repository** bằng:

```python
from ucimlrepo import fetch_ucirepo
covertype = fetch_ucirepo(id=31)
```

Nguồn dữ liệu: [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype).

| Thành phần | Nội dung |
|---|---|
| Số quan sát | 581.012 |
| Biến đầu vào | 54 |
| Biến mục tiêu | `Cover_Type`, nhãn từ 1 đến 7 |
| Biến số | 10 biến địa hình, khoảng cách và độ che sáng |
| Biến nhị phân | 4 cột `Wilderness_Area` và 40 cột `Soil_Type` |
| Kiểu dữ liệu khi nạp | Đầu vào `float32`, nhãn `int8` |
| Kiểm tra đã lưu | 0 ô thiếu, 0 dòng trùng |

Mười biến số gồm `Elevation`, `Aspect`, `Slope`, `Horizontal_Distance_To_Hydrology`, `Vertical_Distance_To_Hydrology`, `Horizontal_Distance_To_Roadways`, `Hillshade_9am`, `Hillshade_Noon`, `Hillshade_3pm` và `Horizontal_Distance_To_Fire_Points`.

| Nhãn | Lớp phủ rừng | Số mẫu Test |
|---|---|---:|
| 1 | Spruce/Fir | 42.368 |
| 2 | Lodgepole Pine | 56.661 |
| 3 | Ponderosa Pine | 7.151 |
| 4 | Cottonwood/Willow | 549 |
| 5 | Aspen | 1.899 |
| 6 | Douglas-fir | 3.473 |
| 7 | Krummholz | 4.102 |

Phân bố Test cho thấy hai lớp đầu chiếm phần lớn quan sát, trong khi Cottonwood/Willow có ít mẫu. Vì vậy, Accuracy được báo cáo cùng Macro-F1 và Balanced Accuracy để tránh che khuất kết quả ở lớp thiểu số.

## 3. Hai notebook đã triển khai

| Notebook | Vai trò |
|---|---|
| [LR_RF_Covertype.ipynb](LR_RF_Covertype.ipynb) | EDA, tiền xử lý, huấn luyện và tối ưu LR/RF với toàn bộ 54 đặc trưng; đánh giá và lưu kết quả. |
| [LR_RF_Covertype_FeatureSelection.ipynb](LR_RF_Covertype_FeatureSelection.ipynb) | Bổ sung `SelectKBest(mutual_info_classif, k=20)` trong pipeline; lặp lại đánh giá LR/RF và đối chiếu kết quả lịch sử của bản 54 biến. |

Trong cả hai notebook, hậu tố **`before`/`after` là trước/sau tối ưu siêu tham số**. Riêng notebook FeatureSelection, cả bốn mô hình đều dùng bộ chọn 20 biến; `before` không có nghĩa là chưa chọn đặc trưng.

## 4. Thiết kế thực nghiệm

### Chia tập và đánh giá chéo

1. Kiểm tra kích thước, nhãn, dữ liệu hữu hạn và cấu trúc các cột nhị phân.
2. Chia Train/Test theo tỷ lệ **80/20**, `stratify=y`, `random_state=42`.
3. Sử dụng **464.809 mẫu Train** và **116.203 mẫu Test**; EDA sau khi tách tập dùng Train.
4. Tạo **StratifiedKFold 3-fold** trên Train, dùng chung các fold cho LR và RF trong mỗi thực nghiệm.
5. Đánh giá mô hình cơ sở, chạy GridSearchCV trên toàn bộ Train, chọn cấu hình theo **CV Macro-F1**, rồi refit trên Train.
6. Đánh giá Test bằng Accuracy, Balanced Accuracy, Precision/Recall/F1 macro, classification report và ma trận nhầm lẫn.

Thực nghiệm thực tế dùng **3-fold**, dù hình thiết kế được nhắc trong notebook gốc ghi 5-fold. Test không được dùng để chọn siêu tham số hoặc điều chỉnh `TOP_K`.

### Pipeline

- **LR với 54 biến:** điền thiếu bằng median và chuẩn hóa `StandardScaler` cho 10 biến số; điền thiếu bằng giá trị phổ biến cho 44 biến nhị phân; sau đó phân loại bằng LR (`lbfgs`, `max_iter=1000`). Các cột nhị phân đã được mã hóa nên không one-hot lại.
- **RF với 54 biến:** phân loại trực tiếp trên dữ liệu đã kiểm tra, không có bước chuẩn hóa.
- **Bản 20 biến:** đặt bộ chọn MI trước các bước trên. Selector được fit bên trong mỗi fold, tránh dùng nhãn của tập validation để chọn biến. Mask phân biệt 10 biến liên tục và 44 biến rời rạc; MI dùng `n_neighbors=3`, `random_state=42`.
- Bản chọn biến sử dụng `joblib.Memory` để cache các bước trong pipeline. `GridSearchCV(n_jobs=1)` phối hợp với ngân sách hai luồng cho các bước tính toán.

### Không gian tìm kiếm

| Mô hình | Siêu tham số khảo sát | Số cấu hình |
|---|---|---:|
| LR | `C ∈ {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}`; `class_weight ∈ {None, balanced}` | 16 |
| RF | `n_estimators ∈ {100, 200}`; `max_depth ∈ {20, None}`; `max_features ∈ {sqrt, 0.5}`; `min_samples_leaf ∈ {1, 3}` | 16 |

Mỗi lưới thực hiện 48 lượt fit CV, sau đó refit cấu hình tốt nhất. LR cơ sở dùng `C=1`, `class_weight=None`; RF cơ sở dùng 100 cây, `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=1`, `class_weight=None`.

## 5. Kết quả thực nghiệm

Nguồn trực tiếp là bảng `comparison` tại cell chỉ số 44 (đếm từ 0) của từng notebook. Các báo cáo lớp nằm ở cell 39–40. CV trong bảng là trung bình ± độ lệch chuẩn của ba fold; các chỉ số còn lại tính trên Test.

### Toàn bộ 54 đặc trưng

| Mô hình | Accuracy | Balanced Accuracy / Recall macro | Precision macro | Macro-F1 | CV Macro-F1 |
|---|---:|---:|---:|---:|---:|
| LR cơ sở | 0,7233 | 0,5089 | 0,5910 | 0,5300 | 0,5325 ± 0,0043 |
| LR tối ưu | 0,7235 | 0,5103 | 0,5909 | 0,5312 | 0,5343 ± 0,0039 |
| RF cơ sở | 0,9538 | 0,9064 | 0,9466 | 0,9249 | 0,9117 ± 0,0003 |
| **RF tối ưu** | **0,9681** | **0,9337** | **0,9541** | **0,9435** | **0,9328 ± 0,0008** |

Cấu hình tốt nhất của LR: `C=100`, `class_weight=None`. RF: `n_estimators=200`, `max_depth=None`, `max_features=0.5`, `min_samples_leaf=1`.

### Chọn 20 đặc trưng bằng Mutual Information

| Mô hình | Accuracy | Balanced Accuracy / Recall macro | Precision macro | Macro-F1 | CV Macro-F1 |
|---|---:|---:|---:|---:|---:|
| LR cơ sở | 0,7168 | 0,4753 | 0,5557 | 0,5002 | 0,5065 ± 0,0054 |
| LR tối ưu | 0,7166 | 0,4767 | 0,5549 | 0,5012 | 0,5076 ± 0,0047 |
| RF cơ sở | 0,9484 | 0,8959 | 0,9424 | 0,9169 | 0,9070 ± 0,0031 |
| **RF tối ưu** | **0,9637** | **0,9252** | **0,9494** | **0,9368** | **0,9264 ± 0,0015** |

Cấu hình tốt nhất của LR: `C=1000`, `class_weight=None`. RF chọn cùng cấu hình tốt nhất như bản 54 biến. Trong từng notebook, RF được chọn theo CV Macro-F1.

### Chất lượng theo từng lớp của RF tối ưu

Các giá trị sau được chép từ classification report, làm tròn hai chữ số như output gốc.

| Lớp | Recall — 54 biến | F1 — 54 biến | Recall — 20 biến | F1 — 20 biến |
|---|---:|---:|---:|---:|
| Spruce/Fir | 0,96 | 0,97 | 0,96 | 0,96 |
| Lodgepole Pine | 0,98 | 0,97 | 0,98 | 0,97 |
| Ponderosa Pine | 0,97 | 0,96 | 0,97 | 0,96 |
| Cottonwood/Willow | 0,88 | 0,89 | 0,86 | 0,88 |
| Aspen | 0,86 | 0,90 | 0,84 | 0,89 |
| Douglas-fir | 0,92 | 0,94 | 0,91 | 0,93 |
| Krummholz | 0,96 | 0,97 | 0,96 | 0,97 |

RF đạt kết quả tốt trên cả các lớp ít mẫu. LR tối ưu có Recall Aspen khoảng 0,01 ở bản 54 biến và làm tròn thành 0,00 ở bản 20 biến, cho thấy hạn chế của LR trong thiết lập này dù Accuracy tổng thể vẫn trên 71%.

### Thời gian đã ghi nhận

Đơn vị: giây. `Fit/refit` là fit mô hình cơ sở hoặc refit cấu hình tối ưu trên toàn bộ Train; `Predict` là trung vị ba lần dự đoán trên 116.203 mẫu Test. Tổng GridSearch là thời gian tìm kiếm và refit, được báo một lần cho mỗi thuật toán.

| Bộ biến | Mô hình | Fit/refit | Predict |
|---|---|---:|---:|
| 54 | LR cơ sở | 89,9480 | 0,2277 |
| 54 | LR tối ưu | 103,2244 | 0,2350 |
| 54 | RF cơ sở | 167,8231 | 4,7968 |
| 54 | RF tối ưu | 789,6306 | 7,8812 |
| 20 | LR cơ sở | 32,7235 | 0,0334 |
| 20 | LR tối ưu | 58,7583 | 0,0337 |
| 20 | RF cơ sở | 81,2366 | 2,1531 |
| 20 | RF tối ưu | 308,3845 | 3,3303 |

| Bộ biến | Tổng GridSearch LR | Tổng GridSearch RF |
|---|---:|---:|
| 54 | 2.637,7950 | 12.719,5829 |
| 20 | 942,7824 | 5.336,2876 |

Các thời gian thuộc hai lần chạy khác nhau; bản MI có cache và chưa xác thực tính đồng nhất môi trường giữa hai lần. Vì vậy, các giá trị này chỉ mô tả lần chạy đã lưu, chưa đủ để kết luận mức tăng tốc do riêng việc chọn biến.

## 6. Phân tích và kết quả đạt được

- **RF vượt LR trong cả hai thực nghiệm.** Với 54 biến, Macro-F1 của RF tối ưu đạt 0,9435 so với 0,5312 của LR tối ưu. Kết quả phù hợp với khả năng của RF trong biểu diễn quan hệ phi tuyến và tương tác giữa các biến; đây là nhận xét từ thực nghiệm, không phải chứng minh cơ chế nhân quả.
- **Tối ưu RF mang lại cải thiện rõ hơn LR.** Với 54 biến, Macro-F1 tăng từ 0,9249 lên 0,9435; với 20 biến, tăng từ 0,9169 lên 0,9368. LR chỉ tăng nhẹ Macro-F1, và Accuracy bản 20 biến giảm nhẹ sau tối ưu vì tiêu chí chọn là CV Macro-F1.
- **Giảm 62,96% số đặc trưng**, từ 54 xuống 20. Khi đối chiếu hai lần chạy, Accuracy RF tối ưu giảm khoảng **0,44 điểm phần trăm**, Macro-F1 giảm khoảng **0,0067**. Bản 20 biến giữ chất lượng cao nhưng không cải thiện điểm dự đoán.
- **Hoàn thiện quy trình thực nghiệm:** kiểm tra dữ liệu, EDA trên Train, pipeline, CV chung trong mỗi notebook, GridSearchCV, đánh giá từng lớp, đo thời gian và mã xuất kết quả để truy vết.

Hai mươi biến được selector chọn khi fit toàn bộ Train trong output đã lưu:

```text
Elevation, Aspect, Slope,
Horizontal_Distance_To_Hydrology,
Horizontal_Distance_To_Roadways,
Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
Horizontal_Distance_To_Fire_Points,
Wilderness_Area1, Wilderness_Area4,
Soil_Type2, Soil_Type4, Soil_Type10, Soil_Type12,
Soil_Type22, Soil_Type23, Soil_Type29, Soil_Type38, Soil_Type39
```

Trong bảng MI đã lưu, `Elevation` có điểm cao nhất (0,458308), tiếp theo là `Wilderness_Area4` (0,146734) và `Horizontal_Distance_To_Roadways` (0,088832). MI là tiêu chí liên hệ đơn biến với nhãn; không đồng nhất với importance của RF và không chứng minh quan hệ nhân quả. Bộ biến có thể khác ở từng fold vì selector được fit riêng.

## 7. Cấu trúc repository

```text
covertype_classification/
├── LR_RF_Covertype.ipynb
├── LR_RF_Covertype_FeatureSelection.ipynb
├── readme.md
├── requirements.txt
└── .gitignore
```

Repository chứa notebook cùng output đã lưu. Dữ liệu được tải khi chạy; các file mô hình và thư mục kết quả chưa có sẵn trong bản nguồn, được tạo bằng các cell phụ lục:

```text
Covertype_RF_LR/runs/<run_id>/       # Bản 54 biến
Covertype_RF_LR_MI/runs/<run_id>/    # Bản chọn 20 biến
```

Các tệp xuất gồm:

- `current_run_results.csv`, `same_experiment_comparison.csv`: chỉ số và thời gian.
- `lr_grid_search_results.csv`, `grid_search_results.csv`: kết quả tìm kiếm LR và RF.
- `{lr,rf}_{before,after}.joblib`: bốn pipeline và metadata đi kèm.
- `*_predictions.csv`, `*_classification_report.csv`: dự đoán và báo cáo từng lớp.
- `environment.json`, `data_manifest.json`: môi trường, cấu hình và nguồn dữ liệu.
- `shared_split.npz`, `train_cv_folds.csv`: chỉ số chia tập và fold CV.
- `lr_best_params.json`, `best_params.json`, `selected_model_by_cv.json`, `comparison_status.json`: tham số và trạng thái hoàn thành.
- Riêng bản MI có thêm `mi_feature_scores.csv`, `rf_selected_feature_importance.csv`, `historical_vs_feature_selection.csv` và cache pipeline.

Mỗi thư mục gốc kết quả có `latest_run.json` trỏ tới lần chạy gần nhất. Notebook còn tính SHA-256 cho dữ liệu, split và fold để hỗ trợ đối chiếu.

## 8. Cài đặt và chạy

### Chạy tại máy cá nhân

Yêu cầu Python và kết nối Internet để cài thư viện, tải dữ liệu UCI. `requirements.txt` liệt kê thư viện phục vụ notebook; đây là khai báo phụ thuộc, chưa phải bản khóa đầy đủ môi trường gốc.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

Nếu không kích hoạt môi trường trên PowerShell, có thể dùng trực tiếp `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` rồi `.\.venv\Scripts\python.exe -m jupyter lab`.

Mở notebook và chạy lần lượt từ trên xuống. Nên đọc bản 54 biến trước, sau đó bản FeatureSelection. Hai notebook chạy độc lập; bảng lịch sử trong bản FeatureSelection là snapshot đã ghi trong mã, không tự đọc kết quả mới của notebook thứ nhất.

### Chạy trên Google Colab

Tải một trong hai notebook lên Colab, cài các thư viện theo cell hướng dẫn và chạy toàn bộ cell theo thứ tự. Các output gốc ghi đường dẫn dưới `/content/`; khi chạy tại máy cá nhân, thư mục kết quả được tạo tương đối theo thư mục làm việc hiện tại. Cần tải thư mục kết quả xuống trước khi kết thúc phiên Colab nếu muốn lưu lâu dài.

### Môi trường trong output gốc

Metadata notebook ghi Python **3.13.18**; các cell môi trường in scikit-learn **1.6.1**, pandas **2.2.3**, NumPy **2.4.6**. Đây là thông tin ghi nhận trong notebook, chưa được kiểm chứng bằng cài đặt lại. `scikit-learn>=1.5` trong requirements đáp ứng API `n_jobs` của `mutual_info_classif`; chạy bằng phiên bản khác có thể thay đổi số liệu hoặc thời gian. Để tái lập chặt chẽ, cần giữ `environment.json`, fingerprint và phiên bản đầy đủ của từng lần chạy.

Toàn bộ GridSearch có thể chạy nhiều giờ; số liệu thời gian ở trên giúp ước lượng từ thực nghiệm cũ. Không thay seed, split, số fold, lưới tham số hoặc `TOP_K` nếu muốn đối chiếu theo cùng thiết kế.

## 9. Giới hạn và hướng phát triển

- Kết luận dựa trên một seed và một split ngẫu nhiên, chưa có kiểm định qua nhiều lần lặp hoặc khoảng tin cậy.
- Split phân tầng ngẫu nhiên chưa đánh giá khả năng tổng quát sang vùng địa lý mới. Có thể bổ sung spatial/group validation khi có thông tin nhóm phù hợp.
- Phụ lục FeatureSelection coi bản 54 biến là kết quả lịch sử, chưa xác thực fingerprint/môi trường tương ứng; đối chiếu hai notebook mang tính mô tả, chưa là benchmark kiểm soát hoàn toàn.
- Holdout cũ được dùng lại để đối chiếu; không tiếp tục chọn `TOP_K` hoặc tinh chỉnh theo Test. Nếu mở rộng chọn biến, nên chọn bằng CV trên Train và đánh giá cuối trên holdout mới độc lập.
- Chưa đo bộ nhớ đỉnh, kích thước mô hình hay hiệu năng phân tán. Đây là hướng bổ sung để đánh giá chi phí xử lý quy mô lớn đầy đủ hơn.
- Có thể khảo sát nhiều số lượng biến, phương pháp chọn biến khác và các thuật toán bổ sung dưới cùng ngân sách đánh giá.

## 10. Tài liệu và nguồn số liệu

- [Notebook thực nghiệm 54 đặc trưng](LR_RF_Covertype.ipynb): mã, biểu đồ, báo cáo lớp, bảng kết quả và phụ lục lưu mô hình.
- [Notebook thực nghiệm chọn 20 đặc trưng](LR_RF_Covertype_FeatureSelection.ipynb): pipeline MI, kết quả mới và bảng đối chiếu lịch sử.
- [Trang dữ liệu UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype): nguồn được hai notebook sử dụng qua `fetch_ucirepo(id=31)`.
