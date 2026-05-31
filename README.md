# Skin Cancer Classification and Analysis Using HAM10000 & DERM12345 Datasets
# Project Overview
This project focuses on Skin Cancer Classification and Analysis using two large-scale dermatoscopic image datasets:

1. HAM10000 Dataset
2. DERM12345 Dataset

The project aims to develop a data-driven and scalable medical imaging pipeline for skin lesion classification using machine learning and deep learning approaches. The study also explores the implementation of big data concepts in healthcare imaging analytics.

# Datasets
1. HAM10000 Dataset

The HAM10000 (“Human Against Machine with 10000 training images”) dataset is one of the most widely used public datasets for pigmented skin lesion analysis.

Dataset Characteristics
- More than 10,000 dermatoscopic images
- Multiple lesion categories
- Expert-labeled annotations
- Widely used for benchmarking skin cancer classification models

2. DERM12345 Dataset

DERM12345 is a large-scale dermatoscopic image dataset containing 12,345 high-resolution images categorized into:

- 5 super classes
- 15 main classes
- 40 subclasses

The dataset was collected in Turkiye and contains diverse skin lesion variations from different skin types.

Dataset Characteristics
- 12,345 high-resolution images
- 40 subclasses of skin lesions
- Expert annotations
- Diverse skin types
- Large hierarchical classification structure

## Dataset License and Citation
This project uses publicly available datasets for educational, research, and portfolio purposes.

### Dataset Licenses
- HAM10000 Dataset — CC BY-NC 4.0
- DERM12345 Dataset — CC BY 4.0

Please cite the original dataset creators and publications when using these datasets.

### HAM10000
Tschandl, P., Rosendahl, C., & Kittler, H. (2018).  
*The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*.  
Scientific Data, 5, 180161.  
https://doi.org/10.1038/sdata.2018.161

### DERM12345
Imperial College London. (2025).  
*DERM12345* [Data set]. ISIC Archive.  
https://doi.org/10.34970/705541

# Objectives of Analysis
The objectives of this project are:

1. To classify skin lesions using machine learning and deep learning methods.
2. To compare the performance of different classification models.
3. To analyze the effectiveness of large-scale dermatoscopic image datasets.
4. To implement big data concepts in medical image analytics.
5. To support early skin cancer detection using AI-based systems.
6. To explore scalable image-processing pipelines for healthcare applications.

# Big Data Characteristics (5V + 3V)
This project implements the concept of Big Data through the following 8V characteristics:

1. Volume

The datasets contain more than 22,000 high-resolution dermatoscopic images combined.

2. Velocity

Medical imaging data continues to grow rapidly due to increasing dermatology imaging systems and healthcare digitization.

3. Variety

The project involves multiple image categories, lesion subclasses, metadata, annotations, and image formats.

4. Veracity

The datasets are annotated and validated by dermatology experts, improving data reliability.

5. Value

The project contributes to early skin cancer detection and AI-assisted healthcare diagnostics.

6. Variability

Skin lesions vary significantly in color, texture, shape, and size across patients and skin types.

7. Visualization

The project utilizes visualization techniques such as:

- Class distribution charts
- Image augmentation previews
- Confusion matrix
- ROC-AUC curves
- Feature importance visualization

8. Vulnerability

Medical image datasets involve healthcare-related information that requires responsible data handling and ethical AI considerations.

# Skin Cancer Classification - Modeling & Evaluation

## Deskripsi
Klasifikasi 7 jenis lesi kulit menggunakan deep learning (transfer learning) pada dataset HAM10000 (10.015 gambar dermoskopi). Menggunakan 2 model: DenseNet201 dan MobileNetV2 dengan two-phase transfer learning.

## Dataset
- **Nama**: HAM10000 (Human Against Machine with 10000 Training Images)
- **Total Gambar**: 10.015 citra dermoskopi
- **Resolusi**: 600 x 450 piksel (RGB), di-resize ke 224 x 224
- **Jumlah Kelas**: 7
- **Pembagian**: 70% training (7.014), 30% validation (3.001)

| Kelas | Jumlah | Persentase |
|-------|--------|-----------|
| Melanocytic Nevi | 6.705 | 66.9% |
| Melanoma | 1.113 | 11.1% |
| Benign Keratosis | 1.099 | 11.0% |
| Basal Cell Carcinoma | 514 | 5.1% |
| Actinic Keratosis | 327 | 3.3% |
| Vascular Lesion | 142 | 1.4% |
| Dermatofibroma | 115 | 1.1% |

## Alur Modeling

### 1. Setup & Persiapan Data
- Gambar diorganisir ke folder berlabel (1 folder = 1 kelas) untuk memastikan label akurat
- Data di-copy ke local storage Colab untuk mempercepat I/O saat training
- Pembagian train/validation menggunakan `validation_split=0.3` pada `ImageDataGenerator`

### 2. Preprocessing Citra

#### Data Augmentation (Training)
Augmentasi dilakukan secara real-time saat training untuk mencegah overfitting dan meningkatkan generalisasi model:
- Rotation: 30 derajat
- Zoom: 20%
- Width & Height Shift: 15%
- Shear: 15%
- Brightness: range [0.8 - 1.2]
- Horizontal & Vertical Flip
- Fill Mode: nearest

#### Preprocessing Spesifik per Model
Setiap arsitektur transfer learning membutuhkan fungsi preprocessing yang berbeda. Ini krusial untuk akurasi:

| Model | Preprocessing | Penjelasan |
|-------|--------------|------------|
| DenseNet201 | `densenet.preprocess_input` | Normalisasi sesuai standar ImageNet (mean subtraction) |
| MobileNetV2 | `rescale=1./255` | Normalisasi piksel ke range [0, 1] |

#### Penanganan Class Imbalance
Dataset memiliki ketidakseimbangan kelas yang ekstrem (Melanocytic Nevi = 66.9% vs Dermatofibroma = 1.1%). Ditangani dengan:
- **Class Weights** menggunakan `compute_class_weight('balanced')` dari scikit-learn
- Bobot diberikan saat training agar model memberikan perhatian lebih pada kelas minoritas

| Kelas | Class Weight |
|-------|-------------|
| Actinic Keratosis | 4.38 |
| Basal Cell Carcinoma | 2.78 |
| Benign Keratosis | 1.30 |
| Dermatofibroma | 12.37 |
| Melanocytic Nevi | 0.21 |
| Melanoma | 1.28 |
| Vascular Lesion | 10.02 |

### 3. Arsitektur Model

#### Model 1: DenseNet201 (Best Model)
- **Base**: DenseNet201 pre-trained pada ImageNet (201 layer)
- **Keunggulan**: Dense connectivity pattern memungkinkan feature reuse yang efisien, cocok untuk medical imaging
- **Head Architecture**:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(256, relu)
  - Dropout(0.4)
  - Dense(7, softmax)

#### Model 2: MobileNetV2
- **Base**: MobileNetV2 pre-trained pada ImageNet
- **Keunggulan**: Arsitektur ringan dan cepat, menggunakan inverted residuals dan linear bottlenecks
- **Head Architecture**: Sama dengan DenseNet201

### 4. Strategi Training: Two-Phase Transfer Learning

Kedua model menggunakan strategi two-phase untuk mengoptimalkan transfer learning:

#### Phase 1: Feature Extraction (Frozen Base)
- Semua layer base model dibekukan (non-trainable)
- Hanya head layer yang di-train
- Learning rate: 0.001 (relatif besar karena head layer random)
- Epochs: 10
- Tujuan: Melatih head layer agar bisa mengklasifikasi fitur dari base model

#### Phase 2: Fine-Tuning (Partially Unfrozen)
- Sebagian layer atas base model dibuka (trainable)
  - DenseNet201: 50 layer teratas
  - MobileNetV2: 30 layer teratas
- Learning rate: 0.0001 (lebih kecil agar bobot pre-trained tidak rusak)
- Epochs: 10
- Tujuan: Menyesuaikan fitur base model dengan karakteristik spesifik citra lesi kulit

#### Callbacks
- **EarlyStopping**: Monitor `val_accuracy`, patience=5, restore best weights
- **ReduceLROnPlateau**: Monitor `val_loss`, factor=0.5, patience=3, min_lr=1e-6

### 5. Hasil Training

#### DenseNet201
| Phase | Best Train Acc | Best Val Acc | Keterangan |
|-------|---------------|-------------|------------|
| Phase 1 (Frozen) | 70.67% | 72.71% | Val acc naik konsisten |
| Phase 2 (Fine-tune) | 73.88% | 71.54% | Fine-tuning menstabilkan model |

#### MobileNetV2
| Phase | Best Train Acc | Best Val Acc | Keterangan |
|-------|---------------|-------------|------------|
| Phase 1 (Frozen) | 58.35% | 60.98% | Peningkatan stabil |
| Phase 2 (Fine-tune) | 66.47% | 67.94% | Signifikan improvement |

### 6. Evaluasi

#### Perbandingan Akurasi

| Model | Accuracy | Keterangan |
|-------|----------|------------|
| **DenseNet201** | **~72.71%** | Best Model |
| MobileNetV2 | ~67.94% | - |

#### Metrik yang Digunakan
- **Accuracy**: Persentase prediksi benar
- **Precision**: Ketepatan prediksi positif per kelas
- **Recall**: Kemampuan mendeteksi setiap kelas
- **F1-Score**: Harmonic mean dari precision dan recall
- **Confusion Matrix**: Detail prediksi per kelas
- **ROC Curve & AUC**: Performa pada berbagai threshold

#### Visualisasi Evaluasi
- Confusion Matrix kedua model
- ROC Curve per kelas
- Perbandingan akurasi dan F1-Score per kelas
- Prediksi model pada gambar asli dengan confidence score
- Training history (accuracy & loss) untuk kedua phase

### 7. Analisis dan Kesimpulan

#### Mengapa DenseNet201 Lebih Baik?
- Dense connectivity memungkinkan setiap layer menerima input dari semua layer sebelumnya
- Feature reuse yang efisien mengurangi jumlah parameter
- Gradient flow lebih baik sehingga training lebih stabil
- Lebih cocok untuk citra medis yang membutuhkan detail fitur halus

#### Tantangan
1. **Class Imbalance**: Melanocytic Nevi mendominasi 66.9% dataset. Ditangani dengan class weights
2. **Preprocessing Kritis**: Setiap model membutuhkan preprocessing berbeda. Kesalahan preprocessing menyebabkan akurasi drop drastis
3. **Keterbatasan Hardware**: GPU diperlukan untuk training optimal. Tanpa GPU, training sangat lambat

#### Kesimpulan
- DenseNet201 dengan two-phase transfer learning mencapai akurasi terbaik ~72.71%
- Preprocessing yang tepat per model sangat krusial untuk performa
- Class weights efektif menangani ketidakseimbangan data
- Data augmentation membantu mencegah overfitting
- Model berhasil mengklasifikasikan 7 jenis lesi kulit dengan performa yang reasonable mengingat tingkat kesulitan dataset HAM10000



