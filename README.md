# 🏃 Human Activity Recognition (HAR) using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to classify human physical activities based on smartphone sensor data. Using a dataset with **561 features** (accelerometer and gyroscope readings), the model distinguishes between six distinct activities:
1.  **Walking**
2.  **Walking Upstairs**
3.  **Walking Downstairs**
4.  **Sitting**
5.  **Standing**
6.  **Laying**

The final model achieves **94% accuracy** and is deployed as an interactive web application using **Streamlit**.

## 🚀 Key Features
* **Dimensionality Reduction:** Utilized **Principal Component Analysis (PCA)** to compress 561 features down to ~100 while retaining **95% of the variance**, significantly improving training speed.
* **Robust Preprocessing:** Implemented StandardScaler and Label Encoding to handle sensor noise and categorical targets.
* **Model Comparison:** Benchmarked Logistic Regression, Random Forest, and Support Vector Machines (SVM).
* **Hyperparameter Tuning:** Optimized the SVM model using **GridSearchCV** (`C=10`, `gamma=0.001`), resolving complex confusion between "Sitting" and "Standing" postures.
* **Web App Deployment:** Created a user-friendly interface allowing users to upload CSV data or load demo samples for instant predictions.

## 📊 Model Performance
After extensive testing, the **Support Vector Machine (SVM)** with an RBF kernel proved to be the best performing model.

| Model | Accuracy |
| :--- | :--- |
| **SVM (Tuned)** | **93.76%** |
| Logistic Regression | 93.11% |
| Random Forest | 87.85% |

**Key Insight:** The model achieved 100% precision on "Laying" but faced minor challenges distinguishing between "Sitting" and "Standing" due to similar gravitational signatures in static poses.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
* **Deployment:** Streamlit
* **Tools:** Jupyter Notebook

## 📂 Project Structure
```text
├── app.py                   # Streamlit Web Application
├── train.csv                # Training dataset
├── test.csv                 # Testing dataset
├── HAR_Project.ipynb        # Jupyter Notebook with EDA & Training logic
├── models/                  # Saved Model Files (.pkl)
│   ├── har_svm_model.pkl    # The trained SVM brain
│   ├── har_scaler.pkl       # Scaler for normalization
│   ├── har_pca.pkl          # PCA configuration
│   └── har_label_encoder.pkl# Label decoder
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
