# Customer Churn Prediction with Artificial Neural Networks

Predict whether a telecom customer will churn (cancel their service) using a feed-forward Artificial Neural Network built with TensorFlow / Keras.

## Overview

Customer churn is one of the most critical metrics for subscription-based businesses. Identifying at-risk customers early allows companies to take proactive retention actions. This project builds a binary classifier that takes customer account and service information as input and outputs a churn probability.

## Dataset

**IBM Telco Customer Churn** — 7,043 customer records with 21 features.

| Category | Features |
|---|---|
| **Demographics** | Gender, Senior Citizen, Partner, Dependents |
| **Account** | Tenure, Contract type, Payment method, Monthly & Total charges |
| **Services** | Phone, Internet, Online security, Online backup, Device protection, Tech support, Streaming TV & movies |
| **Target** | Churn (Yes / No) |

> Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Model Architecture

```
Input (30 features after one-hot encoding)
  │
  ├─ Dense(64, ReLU)
  ├─ Dropout(0.30)
  │
  ├─ Dense(32, ReLU)
  ├─ Dropout(0.20)
  │
  ├─ Dense(16, ReLU)
  │
  └─ Dense(1, Sigmoid)  →  Churn probability
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | patience = 5, restore best weights |
| Data split | 70% train / 15% validation / 15% test |
| Feature scaling | StandardScaler (fit on train only) |

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | 78.96% |
| **Test Loss** | 0.4464 |

### Decision Threshold Analysis

Lowering the threshold increases recall (catching more churners) at the cost of precision:

| Threshold | Precision | Recall |
|---|---|---|
| 0.3 | 0.515 | 0.748 |
| 0.4 | 0.566 | 0.624 |
| 0.5 (default) | 0.609 | 0.529 |
| 0.6 | 0.661 | 0.405 |
| 0.7 | 0.733 | 0.230 |

> **Business insight:** A lower threshold (e.g., 0.3–0.4) is often preferred for churn prediction because the cost of missing an at-risk customer typically outweighs the cost of a false alarm.

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/seelneas/customer_churn_ann.git
cd customer_churn_ann

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install tensorflow scikit-learn pandas matplotlib seaborn
```

### Usage

Open the notebook and run cells from top to bottom:

```bash
jupyter notebook customer_churn_ann_exercise.ipynb
```

The notebook is structured as a guided exercise with `# TODO` prompts. Each task builds on the previous one:

1. **Load & inspect** the data
2. **Clean** — handle blanks in `TotalCharges`, drop `customerID`
3. **Separate** features and target
4. **Encode** categorical variables (one-hot)
5. **Split** into train / validation / test (70 / 15 / 15)
6. **Scale** numerical features with `StandardScaler`
7. **Build** the ANN
8. **Train** with early stopping
9. **Plot** learning curves
10. **Evaluate** on the test set
11. **Confusion matrix** visualization
12. **Threshold analysis** — precision vs. recall trade-off
13. **Save** the trained model

## Project Structure

```
customer_churn_ANN/
├── customer_churn_ann_exercise.ipynb   # Main notebook (guided exercise)
├── customer_churn_ann.keras            # Saved trained model
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── README.md
└── LICENSE                             # MIT License
```

## Key Concepts Covered

- **Feed-forward neural networks** — architecture, activation functions (ReLU, Sigmoid)
- **Binary classification** — sigmoid output, binary crossentropy loss
- **Regularization** — Dropout layers to reduce overfitting
- **Early stopping** — halt training when validation loss stops improving
- **Feature engineering** — one-hot encoding, feature scaling
- **Data leakage prevention** — fitting the scaler only on training data
- **Model evaluation** — accuracy, precision, recall, confusion matrix
- **Threshold tuning** — adjusting the decision boundary for business needs

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.