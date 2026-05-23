# Skin Cancer Classification and Analysis Using HAM10000 & DERM12345 Datasets
# Project Overview
This project focuses on Skin Cancer Classification and Analysis using two large-scale dermatoscopic image datasets:

1. HAM10000 Dataset
2. DERM12345 Dataset

The project aims to develop a data-driven and scalable medical imaging pipeline for skin lesion classification using machine learning and deep learning approaches. The study also explores the implementation of big data concepts in healthcare imaging analytics.

# Project Title
Big Data-Based Skin Cancer Classification Using HAM10000 and DERM12345 Dermatoscopic Image Datasets

# Datasets
1. HAM10000 Dataset

The HAM10000 (“Human Against Machine with 10000 training images”) dataset is one of the most widely used public datasets for pigmented skin lesion analysis.

Dataset Characteristics
- More than 10,000 dermatoscopic images
- Multiple lesion categories
- Expert-labeled annotations
- Widely used for benchmarking skin cancer classification models

Citation

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161. DOI: https://doi.org/10.1038/sdata.2018.161

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

Citation

Imperial College London. (2025). DERM12345 [Data set]. ISIC Archive. DOI: https://doi.org/10.34970/705541

# License and Citation
This project uses publicly available datasets for educational and research purposes.
Please cite the original dataset authors when using the datasets:

- HAM10000:
Tschandl, P., Rosendahl, C., & Kittler, H. (2018).
The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
Scientific Data, 5, 180161.
https://doi.org/10.1038/sdata.2018.161

- DERM12345:
Imperial College London. (2025).
DERM12345 [Data set]. ISIC Archive.
https://doi.org/10.34970/705541


# Tujuan Analisis
Tujuan analisis adalah membangun model klasifikasi citra kanker kulit yang akurat, membantu deteksi dini, memahami pola visual penyakit, Analisis tidak hanya berfokus pada klasifikasi, tetapi juga pada interpretability model dan robustness terhadap variasi data medis.

# Dataset
Dataset yang digunakan adalah Skin Cancer MNIST: HAM10000, dengan karakteristik:
- 10.015 gambar dermatoskopi
- 7 kelas penyakit kulit
- Metadata tambahan

# Big Data Characteristics (5V + 3V)
1. Volume

Dataset terdiri dari lebih dari 10.000 citra medis beresolusi tinggi.
Meskipun tidak berskala petabyte, ukuran ini cukup besar untuk analisis citra berbasis deep learning.

2. Velocity

Dataset bersifat statis (tidak real-time).
Namun dalam implementasi nyata, data citra medis dapat berkembang menjadi sistem streaming (real-time diagnosis).

3. Variety

Memiliki variasi tinggi:
- 7 kelas penyakit kulit
- Data berupa citra + metadata

4. Veracity

Data berasal dari diagnosis medis ahli, namun tetap memiliki tantangan seperti:
- Noise pada label
- Class imbalance

5. Value

Memiliki nilai tinggi karena dapat digunakan untuk:
- Deteksi dini kanker kulit
- Pengembangan sistem AI di bidang kesehatan

6. Variability

Variasi tinggi pada:
-  Bentuk dan warna lesi
-  Kondisi pencahayaan
-  Perbedaan alat pengambilan gambar
7. Visualization

Dataset berbasis citra memungkinkan:
-  Visualisasi langsung
-  Heatmap (Grad-CAM)
-  Analisis distribusi kelas
  
8. Validity

Data dikumpulkan dari sumber medis terpercaya, namun tetap memerlukan:
-  Validasi model
-  Evaluasi performa yang ketat
