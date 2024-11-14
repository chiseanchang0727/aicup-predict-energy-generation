# aicup-predict-energy-generation

- the score is total absolute error.

## Test record

1. Test 3: L10 baseline

- device:L10
- remove: windspeed

- Total absolute error: 224001.19

2. Baseline (without anyother features)

- device: L10
- config: test_3_L10.json
- Total absolute error: 275307.21


3. with positional encoding

- device: L10
- config: test_4_L10.json
- cross validation days: 2
Total absolute error: 223931.44