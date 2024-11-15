## Test record

1. Baseline (without anyother features)

- device: L10
- config: test_3_L10.json
- cross validation days: 2
- Total absolute error: 275307.21
- Note that if cross validation days = 5 the total AE is around 500000

2. with positional encoding

- device: L10
- config: test_4_L10.json
- cross validation days: 2
- Total absolute error: 223931.44

3. month encoding (Sean/main) <- head

- config: test_5_L10_pe.json
- cross validation days: 2
- Total absolute error: 218893.33
