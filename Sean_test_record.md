## Test record

1. Test 3: L10 baseline

- device:L10
- remove: windspeed

- Total absolute error: 224001.19

2. Baseline (without anyother features)

- device: L10
- config: test_3_L10.json
- cross validation days: 2
- Total absolute error: 275307.21
- Note that if cross validation days = 5 the total AE is around 500000

3. with positional encoding

- device: L10
- config: test_4_L10.json
- cross validation days: 2
Total absolute error: 223931.44