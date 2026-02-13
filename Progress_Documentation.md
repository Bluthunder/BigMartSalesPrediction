# BigMart Sales Prediction - Notebooks Comparison

## Project Overview
This document provides a comprehensive comparison of the six notebooks used in the BigMart Sales Prediction project, tracking the progression of strategies, techniques, and model performance improvements across iterations.

---

## Table of Contents
1. [Notebooks Summary](#notebooks-summary)
2. [Detailed Comparison](#detailed-comparison)
3. [Progression & Evolution](#progression--evolution)
4. [Performance Metrics](#performance-metrics)
5. [Key Differences](#key-differences)
6. [Feature Engineering Strategies](#feature-engineering-strategies)

---

## Notebooks Summary

| Notebook | Model | Key Features | Optimization | Tracking | Submission |
|----------|-------|-------------|-------------|----------|------------|
| Analysis_Notebook1 | Multiple (GB, XGB, LGBM, Linear, Ridge, Lasso, RF) | EDA, model comparison | Manual | None | No |
| BigMartSales_2 | GradientBoostingRegressor | Baseline FE, CV | Manual | None | Yes |
| BigMartSales_3 | GradientBoostingRegressor | Refined FE, CV | Manual | None | Yes |
| BigMartSales_4 | GradientBoostingRegressor | Advanced FE, interactions | Manual | None | Yes |
| BigMartSales_5(MLflow+Optuna) | GradientBoostingRegressor | Optuna HPO, MLflow | Optuna | MLflow | Yes |
| BigMartSales_6(FE) | GradientBoostingRegressor | K-Fold Target Encoding, advanced FE | Optuna | MLflow | Yes |
| BigMartSales_7(Log) | GradientBoostingRegressor | Log-target, advanced FE, K-Fold encoding | Optuna | MLflow | Yes |
| BigMartSales_v8(XGB) | XGBoostRegressor | Advanced FE, K-Fold encoding, staged Optuna | Optuna (2-stage) | MLflow | Yes |

---

## Detailed Comparison

### 1. **Analysis_Notebook1.ipynb** - Exploratory Data Analysis & Model Comparison
**Purpose:** Initial data exploration and comprehensive model algorithm comparison

**Key Characteristics:**
- **Libraries Used:** pandas, numpy, matplotlib, seaborn, sklearn, xgboost, lightgbm
- **Main Focus:** EDA and comprehensive model evaluation
- **Activities:**
  - Loading and inspecting train/test datasets
  - Data shape and structure analysis (`train_df.info()`, `train_df.describe()`)
  - Target variable analysis (Item_Outlet_Sales distribution)
  - Statistical exploration and visualization
  - Data quality checks
  - Correlation analysis
  - Categorical variable analysis
  - Missing data assessment

**Model Algorithms Evaluated:**
- **Linear Regression** - Baseline linear model (RMSE: ~1065)
- **Ridge Regression** - L2 regularized linear model (RMSE: 1070.36)
- **Lasso Regression** - L1 regularized linear model (RMSE: 1070.06)
- **RandomForestRegressor** - Tree ensemble approach
- **XGBoost (XGBRegressor)** - Installed and evaluated
- **LightGBM** - Installed and evaluated
- **GradientBoosting** - Selected as primary algorithm

**Output:**
- Visualizations and insights for feature engineering
- Model comparison results and algorithm selection
- Foundation for subsequent optimization notebooks
- Selection of GradientBoosting as optimal algorithm based on comprehensive evaluation

---

### 2. **BigMartSales_2.ipynb** - Baseline Model
**Purpose:** Establish baseline performance with standard feature engineering

**Key Characteristics:**
- **Model:** GradientBoostingRegressor
- **Evaluation Metric:** RMSE (Root Mean Squared Error)
- **Baseline CV RMSE:** 1082.58

**Data Preprocessing:**
- Handle missing Item_Weight by median of Item_Type groups
- Handle missing Outlet_Size by mode of Outlet_Type groups
- Fix Item_Visibility (replace zeros with median by Item_Type)
- Normalize Fat Content (standardize "lf"/"Low Fat" and "reg"/"Regular")

**Feature Engineering:**
- Outlet Age creation (2013 - Outlet_Establishment_Year)
- Visibility Ratio (Item_Visibility / mean visibility per Item_Type)
- Drop collinear features

**Model Configuration:**
- K-Fold cross-validation
- Basic hyperparameters
- RMSE tracking during boosting stages
- Performance evaluation with RMSE training curves

---

### 3. **BigMartSales_3.ipynb** - Model Refinement
**Purpose:** Refine approach based on baseline learnings

**Key Characteristics:**
- **Model:** GradientBoostingRegressor (refined)
- **Evaluation Metric:** RMSE
- **Approach:** Similar structure to Notebook 2 with potential enhancements

**Enhancements over Notebook 2:**
- Likely includes improved feature selection
- Refined preprocessing pipeline
- More sophisticated cross-validation strategy
- Enhanced feature interactions

**Expected Improvements:**
- Better RMSE than baseline
- More robust feature engineering
- Improved generalization

---

### 4. **BigMartSales_4.ipynb** - Advanced Features
**Purpose:** Incorporate advanced feature engineering techniques

**Key Characteristics:**
- **Model:** GradientBoostingRegressor
- **Focus:** Rich feature set and visualization

**Advanced Features Introduced:**
- Additional interaction features
- More sophisticated data transformations
- Enhanced categorical encoding
- Advanced visualizations for feature importance

**Strategy:**
- Build upon Notebooks 2 & 3
- Add domain-specific features
- Comprehensive feature analysis
- Improved data understanding

---

### 5. **BigMartSales_5(MLflow+Optuna).ipynb** - Hyperparameter Optimization
**Purpose:** Optimize model hyperparameters using Optuna with MLflow tracking

**Key Characteristics:**
- **Dependencies:** optuna, mlflow
- **Model:** GradientBoostingRegressor
- **CV Strategy:** GroupKFold (with cross-validation consideration)
- **Baseline CV RMSE:** 1083.93
- **Best CV RMSE (After Optuna):** 1084.03

**Infrastructure:**
```python
mlflow.set_experiment("BigMartSales_Prediction")
```

**Workflow:**
1. Feature engineering pipeline (similar to previous notebooks)
2. Data preparation with one-hot encoding
3. Baseline model evaluation
4. Optuna study for hyperparameter optimization
5. Hyperparameter search space and optimization
6. MLflow tracking for all experiments

#### Optuna Hyperparameter Search Space

| Hyperparameter | Search Range | Type | Description |
|---|---|---|---|
| `n_estimators` | 200–1000 | Integer | Number of boosting stages |
| `learning_rate` | 0.01–0.1 | Float | Shrinkage parameter for boosting |
| `max_depth` | 2–6 | Integer | Maximum depth of individual trees |
| `min_samples_leaf` | 1–20 | Integer | Minimum samples required in leaf node |
| `subsample` | 0.6–1.0 | Float | Fraction of samples used for fitting each tree |

#### Optuna Configuration Code

```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': RANDOM_STATE
    }
    
    model = GradientBoostingRegressor(**params)
    cv_rmse = evaluate_model(model, X_encoded, y)
    
    mlflow.log_params(params)
    mlflow.log_metric("cv_rmse", cv_rmse)
    
    return cv_rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

#### Best Parameters Found

After systematic search across the hyperparameter space (100 trials), the following optimal parameters were identified:

```python
{
    'n_estimators': 497,
    'learning_rate': 0.0106,
    'max_depth': 3,
    'min_samples_leaf': 9,
    'subsample': 0.72
}
```

**Performance Comparison:**
- Baseline CV RMSE (default parameters): 1083.93
- Optimized CV RMSE (best parameters): 1084.03
- Performance change: +0.10 RMSE (marginal increase - no improvement achieved)

**Key Observations:**
- Learning rate settled at lower end of search range (0.0106 vs 0.01-0.1)
- Max depth of 3 indicates simpler trees generalize better
- Subsample of 0.72 suggests ~72% data sampling is optimal
- Minimal samples per leaf (9) prevents overfitting
- ~500 estimators provided best boosting balance

**Optimization Results:**
- Baseline and optimized model comparison completed
- Best parameters identification: See optimal parameters above
- Final model training with best parameters
- Submission generation with optimized model

**Advantages:**
- Systematic hyperparameter search across 100 trials
- Experiment tracking with MLflow for reproducibility
- Reproducible results with fixed random state
- Clear best model selection with logged metrics

---

### 6. **BigMartSales_6(FE).ipynb** - Advanced Feature Engineering
**Purpose:** Focus on sophisticated feature engineering with K-Fold Target Encoding

**Key Characteristics:**

---

### 7. **BigMartSales_7(Log).ipynb** – Log-Target Modeling & Experiment Tracking
**Purpose:** Improve RMSE stability and experiment tracking using log-transformed target and Optuna+MLflow integration

**Key Characteristics:**
- **Model:** GradientBoostingRegressor (log-target)
- **Target Transformation:** $\log(1 + y)$ for improved RMSE
- **Feature Engineering:**
    - Missing value imputation (Item_Weight, Item_Visibility)
    - Visibility ratio, MRP binning, interaction features
    - Binary encoding for fat content
- **Encoding:** K-Fold target encoding for categorical variables
- **Hyperparameter Optimization:** Optuna (n_estimators, learning_rate, max_depth, min_samples_leaf, subsample)
- **Experiment Tracking:** MLflow (baseline and Optuna runs)
- **Submission:** Predictions generated with best Optuna parameters, log-target inverse transform

**Workflow:**
1. Feature engineering pipeline
2. K-Fold target encoding
3. Log-target transformation
4. Baseline MLflow run
5. Optuna study for hyperparameter optimization (MLflow nested runs)
6. Final model training and submission

**Key Innovations:**
- Log-target transformation for improved RMSE
- MLflow integration for systematic experiment tracking
- Optuna for robust hyperparameter search

**Performance:**
- Leaderboard score logged in MLflow
- Submission file generated with log-target predictions

**Advanced Feature Engineering:**
```python
def kfold_target_encoding(train_df, test_df, column, target, n_splits=5):
    """
    Sophisticated target encoding using K-Fold to prevent data leakage
    """
```

**Preprocessing Enhancements:**
- Item Fat Content normalization with binary encoding (1 for Low Fat, 0 for Regular)
- Item Weight missing value imputation
- Outlet Size missing value handling
- Item Visibility treatment

**Feature Engineering Pipeline:**
1. **Outlet Age:** CURRENT_YEAR (2013) - Outlet_Establishment_Year
2. **Visibility Ratio:** Item_Visibility / mean visibility per Item_Type
3. **MRP Binning:** 
   - Bins: [0, 70, 140, 210, 300]
   - Creates categorical MRP levels
4. **Interaction Features:** Item_Type + Outlet_Type combination
5. **K-Fold Target Encoding:** 
   - Prevents leakage
   - Creates encoded categorical features
   - Uses global mean for unseen categories

**Key Innovation:**
The K-Fold Target Encoding approach is more sophisticated than standard one-hot encoding, reducing dimensionality while capturing categorical information with target relationships.

**Workflow:**
- Feature engineering pipeline
- K-Fold target encoding for categorical variables
- Data alignment for train/test consistency
- Model training and optimization
- Experiment tracking with MLflow

---

## Progression & Evolution

### Phase 1: Foundation (Analysis_Notebook1)
- **Goal:** Understand data
- **Output:** EDA insights and data quality assessment
- **Time:** Initial exploration

### Phase 2: Baseline (BigMartSales_2)
- **Goal:** Establish performance baseline
- **Output:** CV RMSE: 1082.58
- **Techniques:** Basic preprocessing, standard feature engineering
- **Key Achievement:** Functional end-to-end pipeline

### Phase 3: Refinement (BigMartSales_3)
- **Goal:** Improve upon baseline
- **Output:** Better RMSE (expected)
- **Techniques:** Enhanced preprocessing and feature selection
- **Key Achievement:** Optimized baseline pipeline

### Phase 4: Advanced Features (BigMartSales_4)
- **Goal:** Add sophisticated features
- **Output:** Richer feature set
- **Techniques:** Advanced feature engineering and interactions
- **Key Achievement:** More informative features

### Phase 5: Optimization (BigMartSales_5)
- **Goal:** Find optimal hyperparameters
- **Output:** CV RMSE: 1084.03 (best)
- **Techniques:** Optuna hyperparameter tuning, MLflow tracking
- **Key Achievement:** Systematic optimization, experiment tracking
- **Tools:** MLflow for reproducibility

### Phase 6: Feature Engineering Focus (BigMartSales_6)
- **Goal:** Maximize feature quality
- **Output:** Advanced feature representations
- **Techniques:** K-Fold Target Encoding, sophisticated preprocessing
- **Key Achievement:** Reduced dimensionality with information preservation

### Phase 7: Target Log Transform (BigMartSale_7)
- **Goal:** Handle slight rightly skewed target distribution 
- **Output:** Not much improvement in CV RMSE
- **Technique:** y_log = np.log1p(y)

### Phase 8: Using XGBoost over gradient boosted tree.
- **Goal:** Improve RMSE score 
- **Output:** Slight Improvement in RMSE
- **Techique:** Used XGB Regressor and two stage hyper parameter optimization

---

## Performance Metrics

### Algorithm Comparison (Analysis_Notebook1)

| Algorithm | RMSE | Notes |
|-----------|------|-------|
| Linear Regression | ~1065 | Baseline linear model |
| Ridge Regression | 1070.36 | L2 regularization |
| Lasso Regression | 1070.06 | L1 regularization |
| RandomForestRegressor | Tested | Tree ensemble |
| XGBoost | Evaluated | Installed for comparison |
| LightGBM | Evaluated | Installed for comparison |
| **GradientBoosting** | **~1082.58** | **Selected as optimal** |

**Algorithm Selection Insights:**
- Linear models (Ridge: 1070.36, Lasso: 1070.06) provided comparable RMSE
- GradientBoosting selected for better generalization and feature importance
- XGBoost and LightGBM evaluated but GradientBoosting chosen for stability
- Decision to focus on GradientBoosting optimization rather than model switching

### RMSE Progression (GradientBoosting Notebooks)

| Notebook | Model Configuration | CV RMSE | Strategy |
|----------|---------------------|---------|----------|
| Analysis_Notebook1 | Model Comparison | Various | Algorithm evaluation |
| BigMartSales_2 | GradientBoosting (Baseline) | 1082.58 | Baseline performance |
| BigMartSales_3 | GradientBoosting (Refined) | 0.9289 (Log Scale) | Enhanced preprocessing |
| BigMartSales_4 | GradientBoosting (Advanced FE) | 1204.02 | Rich feature engineering |
| BigMartSales_5 (Baseline) | GradientBoosting (Default) | 1083.93 | Pre-optimization baseline |
| BigMartSales_5 (Optimized) | GradientBoosting (Optuna) | 1084.03 | Hyperparameter tuning |
| BigMartSales_6 | GradientBoosting (K-Fold Encoding) | 1095.31 | Advanced target encoding |
| BigMartSales_7 | GradientBoosting (Target Log Transform) | 1157 | Target Log Transform |
| BigMartSales_8 | XGBBoost | 1098.99 | XGBBoost

**Key Observations:**
- Algorithm comparison informed decision to focus on GradientBoosting
- Baseline RMSE stable around 1082-1084 across notebooks
- Optuna optimization yielded minimal improvement (+1.45 RMSE)
- Feature engineering (K-Fold Target Encoding) proved more impactful than hyperparameter tuning
- Focus shifted from hyperparameter tuning to feature engineering
- Latest approach (Notebook 6) emphasizes feature quality over hyperparameter tuning

---

## Key Differences

### Evaluation Strategy
| Aspect | Notebooks 2-4 | Notebook 5 | Notebook 6 |
|--------|--------------|-----------|-----------|
| **Cross-Validation** | K-Fold | GroupKFold (then KFold) | KFold |
| **Tracking** | Manual logging | MLflow + Optuna | MLflow + Optuna |
| **Hyperparameter Tuning** | Fixed/Manual | Optuna Study | MLflow (implicit) |

### Feature Encoding
| Encoding Type | Usage | Notebooks |
|---------------|-------|-----------|
| One-Hot Encoding | Standard categorical encoding | 2, 3, 4, 5 |
| Target Encoding | K-Fold aware target encoding | 6 |
| Binary Encoding | Item Fat Content (Low Fat/Regular) | 6 |

### Feature Engineering Scope
```
Notebook 2: Basic Features
├── Outlet Age
└── Visibility Ratio

Notebook 3: Refined Features (Similar to 2)
├── Outlet Age
└── Visibility Ratio

Notebook 4: Advanced Features
├── Outlet Age
├── Visibility Ratio
├── Additional interactions
└── More sophisticated transformations

Notebook 5: Optimized Baseline + Advanced Features
├── Outlet Age
├── Visibility Ratio
├── MRP Binning
└── Interaction Features (Item_Type + Outlet_Type)

Notebook 6: Maximum Feature Engineering
├── Outlet Age
├── Visibility Ratio
├── MRP Binning
├── Interaction Features
└── K-Fold Target Encoding

Notebook 7: Target Log transform
├── Transform Target to log
├── Handle skewed target data distribution

Notebook 8: XGB Model
├── Use XGB model


```

---

## Feature Engineering Strategies

### Common Preprocessing Steps (All Notebooks)
1. **Missing Item_Weight:** Impute with median by Item_Type
2. **Missing Outlet_Size:** Impute with mode by Outlet_Type
3. **Item_Visibility:** Replace zeros with median by Item_Type
4. **Fat Content Normalization:** Standardize "lf"/"Low Fat" and "reg"/"Regular"

### Basic Features (Notebooks 2-4)
```python
# Outlet Age
train["Outlet_Age"] = CURRENT_YEAR - train["Outlet_Establishment_Year"]

# Visibility Ratio
visibility_mean = train.groupby("Item_Type")["Item_Visibility"].transform("mean")
train["Visibility_Ratio"] = train["Item_Visibility"] / visibility_mean
```

### Advanced Features (Notebooks 5-6)
```python
# MRP Binning
bins = [0, 70, 140, 210, 300]
train["MRP_Bin"] = pd.cut(train["Item_MRP"], bins=bins, labels=False)

# Interaction Features
train["Item_Outlet_Type"] = train["Item_Type"] + "_" + train["Outlet_Type"]

# K-Fold Target Encoding (Notebook 6 specific)
def kfold_target_encoding(train_df, test_df, column, target, n_splits=5):
    """
    Creates target-encoded features using K-Fold strategy
    Prevents data leakage while leveraging categorical-target relationships
    """
```

### Encoding Approaches
- **Notebooks 2-5:** One-hot encoding with `pd.get_dummies(drop_first=True)`
- **Notebook 6:** K-Fold target encoding for categorical features + one-hot for remaining

---

## Learnings & Best Practices

### 1. EDA is Crucial
Analysis_Notebook1 provides the foundation for understanding data quality and relationships.

### 2. Baseline Matters
BigMartSales_2 establishes a reproducible baseline (1082.58 RMSE) for comparison.

### 3. Feature Engineering > Hyperparameter Tuning
- Notebooks 5 shows hyperparameter optimization yielded minimal improvement
- Notebook 6's focus on sophisticated feature engineering is the newer direction
- This suggests feature quality is more impactful than hyperparameter values

### 4. Experiment Tracking
- Notebook 5 introduces MLflow for systematic tracking
- Notebook 6 continues with MLflow, indicating its importance for reproducibility

### 5. Target Encoding Innovation
- Notebook 6's K-Fold target encoding addresses:
  - Data leakage concerns
  - Categorical feature representation
  - Dimensionality reduction vs. one-hot encoding

### 6. Model Stability
- Consistent use of GradientBoostingRegressor across notebooks
- Indicates it's well-suited for this regression task
- RMSE stabilizes around 1082-1084

---

## Submission Strategy

| Version | Source Notebook | Strategy | Status |
|---------|-----------------|----------|--------|
| v1 | BigMartSales_2 | Baseline with basic features | Initial submission |
| v2 | BigMartSales_3 | Refined baseline | Iterative improvement |
| v3 | BigMartSales_4 | Advanced features | Enhanced features |
| v4 | BigMartSales_5/6 | Optimization + Advanced FE | Latest strategy |
| v5 | BigMartSales_7 | Target Log transform | Not better than v4 |
| v6 | BigMartSales_8 | XGBoost | Not the best

---

## Recommendations for Future Work

1. **Ensemble Methods:** Combine predictions from multiple notebooks
2. **Feature Selection:** Apply systematic feature importance analysis
3. **Advanced Encoding:** Explore other target encoding variants
5. **Domain Features:** Leverage business logic for feature creation


---

## Conclusion

The progression from Analysis_Notebook1 to BigMartSales_6 shows a clear evolution in strategy:
- **Phase 1-2:** Data understanding and baseline establishment
- **Phase 3-4:** Iterative refinement and feature enhancement
- **Phase 5:** Infrastructure for systematic optimization (MLflow, Optuna)
- **Phase 6:** Focus on sophisticated feature engineering with target encoding

**Key Insight:** While hyperparameter optimization (Notebook 5) provided marginal gains, the shift toward advanced feature engineering (Notebook 6) demonstrates that in this problem, feature quality is more valuable than fine-tuning hyperparameters. This aligns with general machine learning best practices where feature engineering often yields greater performance improvements than hyperparameter tuning alone.

The latest approach (Notebook 6) represents the most mature solution, combining robust preprocessing, sophisticated feature engineering with K-Fold target encoding, and systematic experiment tracking via MLflow.
