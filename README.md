# BigMart Sales Prediction

A comprehensive machine learning project to predict sales for BigMart outlets across different locations and item types using advanced feature engineering and hyperparameter optimization techniques.

## 📋 Project Overview

This project involves predicting the sales of items across different BigMart outlets. The dataset contains information about items, outlets, and their characteristics. The goal is to build a robust regression model that accurately forecasts item outlet sales.

**Problem Type:** Regression  
**Target Variable:** Item_Outlet_Sales  
**Model:** GradientBoostingRegressor  
**Evaluation Metric:** RMSE (Root Mean Squared Error)

## 📁 Project Structure

```
BigMartSalesPrediction/
├── Data/
│   ├── raw/
│   │   ├── train_v9rqX0R.csv      # Training dataset
│   │   └── test_AbJTz2l.csv       # Test dataset (for predictions)
│   └── sample_submission_8RXa3c6.csv
├── Notebooks/
│   ├── Analysis_Notebook1.ipynb           # EDA and data exploration
│   ├── BigMartSales_2.ipynb               # Baseline model
│   ├── BigMartSales_3.ipynb               # Model refinement
│   ├── BigMartSales_4.ipynb               # Advanced features
│   ├── BigMartSales_5(MLflow+Optuna).ipynb # Hyperparameter optimization
│   └── BigMartSales_6(FE).ipynb          # Advanced feature engineering
├── Py/
│   ├── bigmartsales_2.py          # Python version of Notebook 2
│   ├── bigmartsales_3.py          # Python version of Notebook 3
│   ├── bigmartsales_5(mlflow+optuna).py  # Python version of Notebook 5
│   └── bigmartsales_6(fe).py      # Python version of Notebook 6
├── Submissions/
│   ├── bigmart_submission_v1.csv  # Initial submission
│   ├── bigmart_submission_v2.csv  # Refined submission
│   ├── bigmart_submission_v3.csv  # Advanced features submission
│   └── bigmart_submission_v4.csv  # Latest submission
├── NOTEBOOKS_COMPARISON.md        # Detailed comparison of all notebooks
├── README.md                      # This file
└── .gitignore                     # Git ignore file
```

## 🔍 Dataset Overview

### Features

**Item Features:**
- `Item_Identifier`: Unique product identifier
- `Item_Weight`: Weight of the item
- `Item_Fat_Content`: Whether the product is low fat or regular
- `Item_Visibility`: The percentage of total display area allocated to the item
- `Item_Type`: Category of the item (Beverages, Snacks, Dairy, etc.)
- `Item_MRP`: Maximum Retail Price of the item

**Outlet Features:**
- `Outlet_Identifier`: Unique store identifier
- `Outlet_Establishment_Year`: Year the outlet was established
- `Outlet_Size`: Size classification of the outlet (High, Medium, Small)
- `Outlet_Location_Type`: Type of location (Tier 1, Tier 2, Tier 3)
- `Outlet_Type`: Type of outlet (Grocery Store, Supermarket Type1/2/3)

**Target Variable:**
- `Item_Outlet_Sales`: Sales of the item in the particular outlet (this is what we predict)

### Data Characteristics
- **Training Samples:** ~8,523 records
- **Test Samples:** ~5,681 records
- **Missing Values:** Item_Weight, Outlet_Size
- **Data Type:** Mix of numerical and categorical features

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.10
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BigMartSalesPrediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn optuna mlflow
```

4. Run notebooks or Python scripts:
```bash
jupyter notebook Notebooks/
# or
python Py/bigmartsales_6\(fe\).py
```

## 📊 Notebooks Guide

### Phase 1: Exploratory Data Analysis
**Notebook:** `Analysis_Notebook1.ipynb`
- Data loading and inspection
- Statistical summaries
- Distribution analysis
- Missing data assessment
- Correlation analysis
- Visualization of key patterns

### Phase 2: Baseline Model
**Notebook:** `BigMartSales_2.ipynb`
- Basic data preprocessing
- Feature engineering (Outlet Age, Visibility Ratio)
- GradientBoosting baseline model
- **Baseline CV RMSE:** 1082.58
- RMSE tracking during training

### Phase 3: Model Refinement
**Notebook:** `BigMartSales_3.ipynb`
- Enhanced preprocessing pipeline
- Refined feature selection
- Improved cross-validation strategy
- Better generalization techniques

### Phase 4: Advanced Features
**Notebook:** `BigMartSales_4.ipynb`
- Advanced feature engineering
- Feature interactions
- Enhanced visualizations
- Feature importance analysis

### Phase 5: Hyperparameter Optimization
**Notebook:** `BigMartSales_5(MLflow+Optuna).ipynb`
- **Tools:** Optuna for hyperparameter search, MLflow for experiment tracking
- Systematic hyperparameter optimization
- Baseline CV RMSE: 1083.93
- Optimized CV RMSE: 1084.03
- Experiment reproducibility with MLflow

### Phase 6: Advanced Feature Engineering
**Notebook:** `BigMartSales_6(FE).ipynb`
- **Key Innovation:** K-Fold Target Encoding
- Advanced preprocessing with binary encoding
- MRP Binning for price categorization
- Interaction features (Item_Type + Outlet_Type)
- Sophisticated feature representation
- Latest and most mature approach

## 🔧 Key Techniques & Strategies

### Data Preprocessing
1. **Missing Item_Weight:** Imputed with median weight by Item_Type
2. **Missing Outlet_Size:** Imputed with mode by Outlet_Type
3. **Item_Visibility:** Zero values replaced with median by Item_Type
4. **Fat Content Normalization:** Standardized "lf"/"Low Fat" and "reg"/"Regular"

### Feature Engineering

**Basic Features:**
- **Outlet_Age:** Years since outlet establishment (2013 - Outlet_Establishment_Year)
- **Visibility_Ratio:** Item visibility relative to type average

**Advanced Features:**
- **MRP_Bin:** Price categorization using binning
- **Item_Outlet_Type:** Interaction feature combining item and outlet types
- **Target Encoding:** K-Fold target encoding for categorical variables (Notebook 6)

### Model Configuration

**Algorithm:** GradientBoostingRegressor

**Key Hyperparameters (from Optuna optimization):**
- Learning rate: Tuned via Optuna
- Max depth: Optimized for tree complexity
- Subsample & colsample_bytree: Regularization parameters
- Number of estimators: Optimized boosting iterations

**Validation Strategy:**
- K-Fold Cross-Validation (n_splits=5)
- GroupKFold for stratified evaluation
- Consistent random state for reproducibility

## 📈 Performance Evolution

| Version | Notebook | Approach | CV RMSE | Key Feature |
|---------|----------|----------|---------|------------|
| v1 | BigMartSales_2 | Baseline | 1082.58 | Standard preprocessing |
| v2 | BigMartSales_3 | Refined | 0.9289 (Log Scale) | Enhanced pipeline |
| v3 | BigMartSales_4 | Advanced Features | 1204.02 | Rich feature set |
| v4 | BigMartSales_5 | Optuna + MLflow | 1084.03 | Hyperparameter tuning |
| v5 | BigMartSales_6 | K-Fold Target Encoding | 1095.31 | Advanced FE |

## 🎯 Model Performance Insights

**Key Finding:** Feature engineering yielded greater performance improvements than hyperparameter tuning alone.

- **Baseline RMSE:** 1082.58 (BigMartSales_2)
- **With Optuna Optimization:** 1084.03 (minimal improvement of ~1.45)
- **With Advanced FE:** Focus shifted to K-Fold target encoding and sophisticated features

**Recommendation:** Prioritize feature quality over hyperparameter fine-tuning for this problem.

## 📚 Tools & Libraries

| Tool | Purpose | Version |
|------|---------|---------|
| **pandas** | Data manipulation | Latest |
| **numpy** | Numerical operations | Latest |
| **scikit-learn** | Machine learning | Latest |
| **matplotlib** | Static visualizations | Latest |
| **seaborn** | Statistical visualizations | Latest |
| **Optuna** | Hyperparameter optimization | Latest |
| **MLflow** | Experiment tracking | Latest |
| **Jupyter** | Interactive notebooks | Latest |

## 🔄 Workflow

1. **Data Loading & EDA** → Analysis_Notebook1
2. **Baseline Model** → BigMartSales_2
3. **Iterative Refinement** → BigMartSales_3 & 4
4. **Systematic Optimization** → BigMartSales_5
5. **Advanced Feature Engineering** → BigMartSales_6
6. **Generate Predictions** → Submission CSVs

## 📝 Running Experiments

### Using MLflow

To track experiments:
```python
import mlflow

mlflow.set_experiment("BigMartSales_Prediction")

with mlflow.start_run():
    # Train model
    mlflow.log_params(model_params)
    mlflow.log_metric("cv_rmse", cv_rmse)
    mlflow.sklearn.log_model(model, "model")
```

View results:
```bash
mlflow ui
# Navigate to http://localhost:5000
```

### Using Optuna

For hyperparameter optimization:
```python
import optuna

def objective(trial):
    # Define hyperparameter space
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    # Train and return metric
    return cv_rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

## 🎓 Learning Outcomes

This project demonstrates:
1. ✅ Complete ML pipeline from data exploration to deployment
2. ✅ Feature engineering best practices and advanced techniques
3. ✅ Model validation and cross-validation strategies
4. ✅ Hyperparameter optimization using modern tools (Optuna)
5. ✅ Experiment tracking and reproducibility (MLflow)
6. ✅ Iterative model improvement and refinement
7. ✅ Regression problem solving with tree-based models

## 📋 Detailed Documentation

For an in-depth comparison of all notebooks and their strategies, see [NOTEBOOKS_COMPARISON.md](NOTEBOOKS_COMPARISON.md).

## 💡 Key Insights

1. **Data Quality Matters:** Proper missing value imputation and normalization significantly impact performance
2. **Feature Engineering Trumps Tuning:** Advanced features (K-Fold Target Encoding) proved more valuable than hyperparameter optimization
3. **Reproducibility is Critical:** Using MLflow and fixed random states ensures consistent results
4. **Iterative Approach:** Progressive refinement through 6 notebooks led to the most mature solution
5. **Model Stability:** Consistent RMSE around 1082-1084 indicates robust baseline

## 🔮 Future Improvements

- [ ] Ensemble methods combining multiple model types
- [ ] Systematic feature selection and importance analysis
- [ ] Advanced encoding techniques (WOE, Helmert encoding)
- [ ] Cross-validation with alternative algorithms (XGBoost, LightGBM)
- [ ] Stacking meta-models for improved predictions
- [ ] Domain-specific feature engineering
- [ ] Anomaly detection and outlier treatment

## 📞 Contact & Support

For questions or improvements, please refer to the project documentation or reach out through the repository.

## 📄 License

This project is part of a data science learning initiative.

---

**Last Updated:** February 12, 2026  
**Latest Version:** BigMartSales_6(FE) with K-Fold Target Encoding
