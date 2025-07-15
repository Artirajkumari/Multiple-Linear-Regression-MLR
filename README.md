# Multiple Linear Regression (MLR) with Feature Selection

This project demonstrates the implementation of Multiple Linear Regression using both Scikit-learn and Statsmodels.

## 📌 Objective
To identify the most impactful features for predicting company profits using a real-world dataset.

## 🔧 Tools Used
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- Statsmodels

## 📊 Dataset
- **50_Startups.csv**: Contains R&D spend, administration cost, marketing spend, and profit for 50 startups.

## ✅ Steps Performed
1. Data loading and preprocessing
2. One-hot encoding for categorical features
3. Train-test split
4. Linear regression model training (Scikit-learn)
5. Model evaluation using `.score()`
6. Feature selection using OLS summary from Statsmodels
7. Iterative elimination of features with high p-values (> 0.05)
8. Identified R&D spend as the most significant predictor of profit

## 📈 Conclusion
- R&D investment is the most impactful for maximizing profits.
- Feature selection using p-values helps refine the regression model.


