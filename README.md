# 🩺 Diabetes Prediction using Support Vector Machine (SVM)

A machine learning project for predicting diabetes outcomes using the **Pima Indians Diabetes Dataset**.
The project focuses on data cleaning, preprocessing, feature scaling, Support Vector Machine classification, hyperparameter tuning, robustness testing with noisy data, and model comparison.

---

## 📌 Project Overview

This project builds and evaluates several **SVM classifiers** for binary diabetes prediction.

The complete workflow includes:

* Exploratory Data Analysis (EDA)
* Missing and invalid-value detection
* Median-based data imputation
* Feature/target separation
* Stratified train-test splitting
* Feature standardization with `StandardScaler`
* SVM training with multiple kernels
* Model evaluation using accuracy and classification reports
* Hyperparameter tuning using `GridSearchCV`
* Support vector analysis
* Robustness testing by adding noise to the `Glucose` feature
* Comparison with Logistic Regression and Random Forest

---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

**Dataset source:**
[Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The dataset contains **768 samples** and **8 input features**.

### Features

| Feature                    | Description                  |
| -------------------------- | ---------------------------- |
| `Pregnancies`              | Number of pregnancies        |
| `Glucose`                  | Plasma glucose concentration |
| `BloodPressure`            | Diastolic blood pressure     |
| `SkinThickness`            | Triceps skin fold thickness  |
| `Insulin`                  | 2-Hour serum insulin         |
| `BMI`                      | Body mass index              |
| `DiabetesPedigreeFunction` | Diabetes pedigree function   |
| `Age`                      | Age of the patient           |
| `Outcome`                  | Target variable              |

### Target

* `0` → No diabetes
* `1` → Diabetes

---

## 🔍 Exploratory Data Analysis

The notebook performs several exploratory analysis steps:

### Target Distribution

The dataset contains:

* **65.10%** → No diabetes
* **34.90%** → Diabetes

### Feature Correlation

The strongest positive correlation with the target is:

1. `Glucose` → **0.467**
2. `BMI` → **0.293**
3. `Age` → **0.238**
4. `Pregnancies` → **0.222**

`Glucose` is clearly the feature most correlated with the diabetes outcome in this dataset.

---

## 🧹 Data Cleaning

Although the dataset contains no actual `NaN` values, several medical features contain **zero values that are not biologically meaningful**.

The following features were treated as invalid when equal to zero:

```text
Glucose
BloodPressure
SkinThickness
Insulin
BMI
```

### Invalid Zero Values

| Feature       | Invalid Zeros |
| ------------- | ------------: |
| Glucose       |             5 |
| BloodPressure |            35 |
| SkinThickness |           227 |
| Insulin       |           374 |
| BMI           |            11 |

The invalid zeros were:

1. Replaced with `NaN`
2. Replaced using the feature's **median**

Median imputation was selected because it is more robust to outliers and skewed distributions than the mean.

---

## ⚙️ Preprocessing

The dataset is split into:

```python
X = features
y = target
```

A stratified train-test split was used:

* **Training:** 614 samples
* **Testing:** 154 samples
* `test_size = 0.2`
* `random_state = 42`

### Feature Scaling

`StandardScaler` was applied before training the SVM models.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted **only on the training data** and then applied to the test set to avoid data leakage.

---

# 🤖 SVM Models

Three different SVM kernels were trained:

### 1. Linear Kernel

```python
SVC(kernel="linear", random_state=42)
```

### 2. RBF Kernel

```python
SVC(kernel="rbf", random_state=42)
```

### 3. Polynomial Kernel

```python
SVC(kernel="poly", degree=3, random_state=42)
```

---

## 📈 Initial SVM Results

| Kernel     | Test Accuracy |
| ---------- | ------------: |
| Linear     |    **70.13%** |
| RBF        |    **74.03%** |
| Polynomial |    **71.43%** |

The initial RBF model achieved the highest test accuracy among the three directly trained SVM variants.

---

# 🎯 Hyperparameter Tuning

`GridSearchCV` was used to optimize:

* `C`
* `gamma`
* `kernel`

### Search Space

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["linear", "rbf", "poly"]
}
```

Five-fold cross-validation was used with accuracy as the scoring metric.

### Best Parameters

```text
C = 0.1
gamma = scale
kernel = linear
```

### Cross-Validation Accuracy

**77.37%**

### Test Set Performance

**70.78% accuracy**

```text
Tuned SVM Accuracy: 70.78%
```

The difference between cross-validation and held-out test performance highlights the importance of evaluating the final model on unseen data.

---

# 🔹 Support Vectors

The tuned SVM used:

* **159 support vectors** for class `0`
* **159 support vectors** for class `1`
* **318 total support vectors**

This represents approximately:

**51.8% of the training dataset**

Support vectors are the training samples closest to the decision boundary and are the critical points used by SVM to define the separating hyperplane.

---

# 🧪 Robustness Test: Noisy Data

To test model robustness, Gaussian noise was added to the `Glucose` feature:

```python
noise = np.random.normal(
    loc=0,
    scale=20,
    size=X_train_noisy["Glucose"].shape
)
```

The model was then retrained using the noisy training data.

### Results

| Condition  |   Accuracy |
| ---------- | ---------: |
| Clean Data | **70.78%** |
| Noisy Data | **73.38%** |

In this particular experiment, adding noise did not reduce accuracy. The result should be interpreted as a single experimental run rather than evidence that noise generally improves the model.

---

# 🆚 Model Comparison

The tuned SVM was compared with two additional classification models.

### Models

* Support Vector Machine
* Logistic Regression
* Random Forest

### Test Accuracy

| Model               |   Accuracy |
| ------------------- | ---------: |
| Tuned SVM           | **70.78%** |
| Logistic Regression | **70.78%** |
| Random Forest       | **74.03%** |

This comparison shows that different algorithms can perform differently on the same preprocessed dataset and that model selection should be based on the evaluation criteria relevant to the application.

---

# 🛠️ Technologies Used

* **Python**
* **Jupyter Notebook**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

### Main Machine Learning Techniques

```text
Exploratory Data Analysis
Data Cleaning
Median Imputation
Feature Scaling
Train/Test Split
Support Vector Machine
Linear Kernel
RBF Kernel
Polynomial Kernel
GridSearchCV
Cross-Validation
Logistic Regression
Random Forest
Model Evaluation
Noise Robustness Testing
```

---

# 📁 Project Structure

```text
.
├── SVM_Notebook_solved.ipynb
├── diabetes.csv
└── README.md
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
SVM_Notebook_solved.ipynb
```

Make sure `diabetes.csv` is located in the same directory as the notebook.

---

# 🎓 Key Learning Outcomes

This project demonstrates practical understanding of:

* Preparing real-world tabular data for machine learning
* Detecting invalid values that are not represented as `NaN`
* Choosing median imputation for skewed features
* Preventing data leakage during preprocessing
* Understanding why feature scaling is important for SVM
* Comparing different SVM kernels
* Performing systematic hyperparameter optimization
* Understanding the role of support vectors
* Testing model robustness under noisy input data
* Comparing SVM with other classical classification algorithms

---

## ⚠️ Disclaimer

This project is intended for **educational and machine learning practice purposes only**.
It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## 👩‍💻 Author

**Shrouk Mohamed**

Software Engineering Graduate | Machine Learning & AI Enthusiast

[GitHub](https://github.com/shroukmohamed5)
