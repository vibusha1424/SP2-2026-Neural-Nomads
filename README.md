# Sona Power Predict – 2026

## College Name
Sona College of Technology

---

## Team Name
Neural Nomads

---

## Team Members

| Name | College Year | Department |
|------|------|------|
| Ela Vibusha | 2nd Year | CSD |
| Niranjana E | 2nd Year | CSD |
| Kaveena M | 2nd Year | CSD |
| Deepak P | 2nd Year | CSD |

---

## Libraries Used

### Machine Learning
- XGBoost
- LightGBM
- Scikit-learn

### Data Processing
- Pandas
- NumPy

### Utilities
- difflib (SequenceMatcher)
- warnings
- collections

---

## Brief Explanation of Our Approach / Model

The Powerplay phase in cricket is highly unpredictable, with scores varying significantly based on venue conditions, team strategy, player form, and match situations.

Our solution uses a **regime-aware ensemble architecture** instead of a single regression model.

### Stage 1 — Regime Classification
A LightGBM classifier (with XGBoost fallback) first predicts the probability of the innings belonging to one of three scoring regimes:

- Low (<44)
- Normal (44–65)
- Explosive (>65)

### Stage 2 — Regime-Specific Regression
Three separate XGBoost regressors are trained independently for each regime.

Instead of forcing a hard classification, the final prediction is generated using a probability-weighted mixture of all three regressors, improving prediction flexibility and stability.

---

## Feature Engineering

The model uses multiple levels of feature engineering:

- Venue-based features
- Team historical performance
- Recent form analysis
- Ball-by-ball player statistics
- Explosion probability estimation

Additional techniques such as:
- James-Stein shrinkage
- Regime-aware calibration
- Temporal cross-validation
- Fuzzy player name matching

are used to improve robustness and prevent overfitting.

---

## Model Highlights

- Regime-aware prediction architecture
- Dedicated models for explosive innings
- Strict temporal leakage prevention
- Deterministic training setup
- Out-of-fold calibration
- Walk-forward validation strategy

---


