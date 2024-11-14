# aicup-predict-energy-generation

- the score is total absolute error.

## Test record

1. test1

- device: L8
- no preprocessing
- no feature engineering
- model: xgboost (no hyperparameter tuning)

- Total absolute error: -202681.06

2. Test 2

- device: L8
- remove windspeed

- Total absolute error -202681.06

3. Test 3: L10 baseline

- device:L10
- remove: windspeed

- Total absolute error: 224001.19

4. Baseline (without anyother features)

- device: L10
- Total absolute error: 520669.06
