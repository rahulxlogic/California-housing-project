# 🏠 California House Price Prediction using Scikit-Learn

## 📌 Project Overview
This project implements an **end-to-end machine learning pipeline** to predict house prices using structured housing data.  
It demonstrates a **real-world ML workflow** — from data preprocessing and feature engineering to model training, evaluation, persistence, and inference.

Although the pipeline uses the **California Housing dataset**, the approach is designed to be **directly transferable to any other city housing data**.

---

## 🎯 Problem Statement
Predict the **median house value** based on multiple numerical and categorical features such as:
- Location (latitude & longitude)
- Median income
- Number of rooms
- Housing age
- Population density
- Proximity to key locations (categorical)

This is a **supervised regression problem**.

---

## 🧠 Key Concepts Implemented
- Exploratory Data Analysis (EDA)
- Stratified train-test split
- Data preprocessing pipelines
- Handling missing values
- Feature scaling & encoding
- Model comparison & evaluation
- Cross-validation
- Model persistence & inference

---

## 🗂 Dataset
- **California Housing Dataset**
- Contains both **numerical** and **categorical** features

---

## 🔍 Exploratory Data Analysis (EDA)
EDA includes:
- Dataset inspection (`head`, `info`, `describe`)
- Correlation analysis
- Geographic visualization
- Feature–target relationship analysis

These steps help identify:
- Important predictors
- Data imbalance
- Outliers and distributions

---

## 🧪 Train-Test Splitting
To avoid data leakage:
- **Stratified sampling** is applied based on income categories
- Ensures consistent income distribution across train and test sets

Tool used:
- `StratifiedShuffleSplit`

---

## ⚙️ Data Preprocessing Pipeline
A **reproducible preprocessing pipeline** is built using Scikit-Learn.

### Numerical Features
- Median imputation
- Standard scaling

### Categorical Features
- One-Hot Encoding
- Handles unseen categories safely

### Tools Used
- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`

---

## 🤖 Models Trained
The following regression models are trained and compared:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## 📊 Model Evaluation
Evaluation metrics:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

To ensure reliable performance estimates:
- **Cross-validation** is used instead of relying only on training metrics

📌 **Random Forest Regressor** achieved the best and most stable performance.

---

## 💾 Model Persistence & Inference
To make the project production-ready:
- Trained model is saved using **joblib**
- Preprocessing pipeline is saved separately
- Inference can be run on new input data without retraining

### Workflow
1. Check if saved model exists  
2. Train & save if not present  
3. Load model and pipeline for prediction  
4. Generate predictions for new data (`input.csv`)
5. Save results to `output.csv`

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `housing.csv` | The dataset used for training and testing. |
| `input.csv` | New data for making predictions (inference). |
| `output.csv` | The file where the model saves its predictions. |
| `main.py` | The primary script containing the pipeline, training logic, and inference system. |
| `model.pkl` | The serialized Random Forest model (generated after training). |
| `pipeline.pkl` | The serialized preprocessing pipeline (generated after training). |



---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
---
## 2️⃣ Run the Script
```
python main.py
```

- *First Run:* The script will train the model, save it to ```model.pkl```, and generate predictions in ```output.csv.```

- *Subsequent Runs:* The script will detect the saved model and skip training, performing only inference.
---
## 🛠 Tech Stack

Python

Pandas, NumPy

Scikit-Learn

Matplotlib

Joblib

---
## 📈 Results

- **Metric:** Root Mean Squared Error (RMSE)

- **Best Model:** Random Forest Regressor

- **Conclusion:** The Random Forest model generalized best to unseen data compared to Linear Regression and Decision Trees.
---
## 🔮 Future Improvements

Replace proxy dataset with real housing data

Hyperparameter tuning (GridSearchCV)

Feature importance visualization

Deploy as a REST API

---

## ✅ Key Takeaway

This project demonstrates how to build a clean, modular, and production-ready machine learning regression pipeline, suitable for real-world deployment and scalable to new datasets.




