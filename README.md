#  House Price Prediction using Multiple Linear Regression

> A machine learning project that predicts house prices from scratch using NumPy — implementing gradient descent, feature normalization, and model evaluation without relying on high-level ML frameworks.

---

##  Problem Statement

Predicting house prices is a classic regression problem in real estate and data science. Given a set of physical characteristics of a house, the goal is to build a model that accurately estimates its market price. This project implements **Multiple Linear Regression from scratch** using NumPy, demonstrating a ground-up understanding of the core machine learning workflow — from raw data ingestion to prediction.

---

## 📂 Dataset Description

The dataset is loaded from `data/houses.txt` and contains residential housing records with the following columns:

| Feature | Description | Type |
|---|---|---|
| `size(sqft)` | Total area of the house in square feet | Continuous |
| `bedrooms` | Number of bedrooms | Discrete |
| `floors` | Number of floors | Discrete |
| `age` | Age of the house in years | Continuous |
| `Price (1000s dollars)` | **Target variable** — sale price in thousands of USD | Continuous |

**Key Observations:**
- Features span vastly different scales (e.g., `size` ≈ 952–1947 sqft vs. `age` ≈ 17–65 years), making **feature normalization essential** for stable gradient descent.
- There are 4 input features (`n = 4`) and the dataset contains multiple training examples (`m` rows).

---

## 🤖 Machine Learning Approach

This project uses **Multiple Linear Regression** — a supervised learning algorithm that models the relationship between multiple input features and a continuous output variable.

The hypothesis function is:

$$\hat{y} = \mathbf{w} \cdot \mathbf{x} + b = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b$$

Where:
- $\mathbf{w} = [w_1, w_2, w_3, w_4]$ — weight vector (one weight per feature)
- $b$ — bias term (scalar)
- $\mathbf{x}$ — feature vector for a single training example
- $\hat{y}$ — predicted house price

---

## 📐 Mathematical Intuition

### Cost Function (Mean Squared Error)

The model is trained by minimizing the **Mean Squared Error (MSE)** cost function:

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

Where $m$ is the number of training examples, $\hat{y}^{(i)}$ is the prediction, and $y^{(i)}$ is the true label.

### Gradient Descent

Model parameters are updated iteratively to minimize $J(\mathbf{w}, b)$:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}, \quad b := b - \alpha \frac{\partial J}{\partial b}$$

The partial derivatives are:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)$$

Where $\alpha$ is the **learning rate**, controlling the step size of each update.

---

## ⚙️ Data Preprocessing

### Feature Normalization (Z-score Standardization)

Raw features are normalized using the **training set mean ($\mu$) and standard deviation ($\sigma$)**:

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

This is critical because:
- Features like `size(sqft)` dominate gradients without normalization
- Normalization ensures that gradient descent converges faster and more reliably
- The same $\mu$ and $\sigma$ computed from training data are applied to new/unseen inputs at inference time

```python
def normalize_data(x):
    mu = np.mean(x, axis=0)      # Column-wise mean
    sigma = np.std(x, axis=0)    # Column-wise standard deviation
    normalized = (x - mu) / sigma
    return mu, sigma, normalized

X_mu, X_sigma, X_normalized = normalize_data(X_train)
```

**Effect on Training:** Before normalization, a small learning rate of `1e-7` was required with only 50 iterations. After normalization, the learning rate was increased to `1e-1` with 1000 iterations, achieving significantly better convergence.

---

##  Model Training

### Initialization

```python
w_init = np.zeros(n)   # Initialize weights to zero
b_init = 0             # Initialize bias to zero
```

### Cost Function Implementation

```python
def cost_function(x, y, w, b, m):
    squared_error = 0
    for i in range(m):
        y_pred = np.dot(w, x[i]) + b
        error = y_pred - y[i]
        squared_error += np.square(error)
    return squared_error / (2 * m)
```

### Gradient Computation

```python
def compute_derivative(x, y, w, b, m, n):
    dj_db = 0
    dj_dw = np.zeros(n)
    for i in range(m):
        y_pred = np.dot(w, x[i]) + b
        error = y_pred - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
    return dj_dw / m, dj_db / m
```

### Gradient Descent Loop

```python
alpha = 1.0e-1      # Learning rate (post-normalization)
iterations = 1000   # Training iterations

final_w, final_b, loss_history, hist_iterations = gradient_descent(
    X_normalized, Y_train, w_init, b_init,
    compute_derivative, cost_function, alpha, iterations
)
```

---

## 📉 Learning Curve

The cost function is tracked at every iteration and plotted to monitor convergence:

```python
plt.plot(hist_iterations, loss_history)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()
```

A steadily decreasing cost curve without oscillation confirms that the learning rate is well-tuned and the model is converging correctly.

---

## 📊 Model Evaluation

After training, predictions are generated for all training examples and compared against actual prices using scatter plots (one per feature). Evaluation metrics used:

| Metric | Formula | Purpose |
|---|---|---|
| **MSE** | $\frac{1}{m}\sum(\hat{y} - y)^2$ | Measures average squared prediction error |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Measures proportion of variance explained (1.0 = perfect) |

---

## 🔮 Prediction on New Data

New inputs are normalized using the **training set statistics** before prediction:

```python
x_house = np.array([1200, 3, 1, 40])               # 1200 sqft, 3 bed, 1 floor, 40 years old
x_house_norm = (x_house - X_mu) / X_sigma          # Normalize using training mu & sigma
x_house_predict = np.dot(x_house_norm, final_w) + final_b

# Output: predicted price of a house with 1200 sqft, 3 bedrooms,
#         1 floor, 40 years old = $318,936
```

---

## 🚀 How to Run the Project

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Clone the Repository

```bash
git clone [https://github.com/AsokTamang/Supervised-Learning]
cd house-price-prediction
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 4. Launch the Notebook

```bash
jupyter notebook House_Price_Prediction.ipynb
```

### 5. Run All Cells

In Jupyter, select **Kernel → Restart & Run All** to execute the full pipeline from data loading to prediction.

---

## 🗂️ Project Structure

```
house-price-prediction/
│
├── data/ folder was in .gitignore as excessive file size , so
 Data link : https://drive.google.com/file/d/1n9ELAOYjLcCApx1m_JimB_9qoWF82Q_p/view?usp=sharing
│   
│
├── House_Price_Prediction.ipynb    # Main Jupyter Notebook
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **NumPy** | Vectorized math, matrix operations, gradient computation |
| **Pandas** | Data loading and DataFrame manipulation |
| **Matplotlib** | Visualization — scatter plots, learning curve |
| **Jupyter Notebook** | Interactive development and visualization |

---

## 📈 Results & Insights

- **Without normalization:** Gradient descent required a very small learning rate (`α = 1e-7`) and converged slowly over just 50 iterations due to the differing feature scales.
- **After Z-score normalization:** The learning rate was increased 1,000,000× to `α = 1e-1`, and the model converged cleanly in 1,000 iterations with a smooth, monotonically decreasing cost curve.
- **Sample prediction:** A house with 1,200 sqft, 3 bedrooms, 1 floor, and 40 years of age was predicted at approximately **$318,936**.
- **Key insight:** Feature `size(sqft)` is the strongest predictor of price; `age` has an inverse relationship with price, while `bedrooms` and `floors` contribute positively.

---

## 🔭 Future Improvements

- **Polynomial features:** Capture non-linear relationships between features and price (e.g., `size²`).
- **Regularization (Ridge/Lasso):** Add L1/L2 penalties to reduce overfitting on larger datasets.
- **Vectorized implementation:** Replace Python loops with fully vectorized NumPy operations for significant speed improvements.
- **Train/Validation/Test split:** Evaluate true generalization performance on held-out data.
- **Additional features:** Incorporate location data, number of bathrooms, garage size, or proximity to amenities.
- **Advanced models:** Compare results against Decision Trees, Random Forest, or XGBoost for benchmarking.
- **Hyperparameter tuning:** Automate the search for optimal learning rate and iteration count.
- **Web deployment:** Wrap the model in a Flask/FastAPI app for real-time price predictions.

---

## 👤 Author

**Asok Tamang**
- GitHub: https://github.com/AsokTamang
- LinkedIn: https://www.linkedin.com/in/asok-tamang11/

---


*Built from scratch to deeply understand the mathematics behind Multiple Linear Regression — no black boxes, just NumPy.*
