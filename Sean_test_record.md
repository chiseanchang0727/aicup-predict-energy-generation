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

3. month encoding (Sean/main)

- config: test_5_L10_pe.json
- cross validation days: 2
- Total absolute error: 218893.33

4. with saturated sunlight value repalcement

- config: test_6_L10_sunlight_sim.json
  - window_length:180, polyorder:8
- cross validation days: 2
- Total absolute error: 216253.72 or 216953.42
- result_file_name = test_6_L10_sunlight_sim_result.csv

5. with saturated sunlight value repalcement

- config: test_7_L10_pe.json
  - window_length:150, polyorder:8
- cross validation days: 2
- Total absolute error: 216253.72
- result_file_name = test_6_L10_sunlight_sim_result.csv

6. with saturated sunlight value repalcement

- config: test_6_L10_sunlight_sim.json
  - window_length:100, polyorder:6
- cross validation days: 2
- Total absolute error: 214905.66
- result_file_name = test_6_L10_sunlight_sim_result.csv

7. with saturated sunlight value repalcement

- config: test_8_L10_sunlight_sim.json
  - window_length:95, polyorder:2
- cross validation days: 2
- Total absolute error: 213837.82
- result_file_name = test_8_L10_sunlight_sim_result.csv

8. with saturated sunlight value repalcement

- remove year, day_of_week and quarter features
- config: test_8_L10_sunlight_sim.json
  - window_length:95, polyorder:2
- cross validation days: 2
- Total absolute error: 188562.32
- result_file_name = test_8_L10_sunlight_sim_result.csv

9. add average sunlight in certain time window (without standardization)

- remove year, day_of_week and quarter features
- config: test_10_L10_grouping_1.json
  - window_length:95, polyorder:2
  - grouping: 7
- cross validation days: 2
- Total absolute error: 179091.56
- result_file_name = test_10_L10_grouping_1_result.csv

10. add average and difference sunlight in certain time window (WITH correct standardization)

- remove year, day_of_week and quarter features
- config: test_10_L10_grouping_1.json
  - window_length:95, polyorder:2
  - grouping: 7
- cross validation days: 2
- Total absolute error: 473.77
- result_file_name = test_10_L10_grouping_1_result.csv

11. based on 10, add average and difference humidity

- remove year, day_of_week and quarter features
- config: test_10_L10_grouping_1.json
  - window_length:95, polyorder:2
  - grouping: 7
- cross validation days: 2
- Total absolute error: 470.14
- result_file_name = test_10_L10_grouping_1_result.csv

12. change n_split(train, test portion) to 8

- remove year, day_of_week and quarter features
- config: test_10_L10_grouping_1.json
  - window_length:95, polyorder:2
  - grouping: 7
- cross validation days: 8
- n_split = 8 
- Total absolute error: 425.13
- result_file_name = test_10_L10_grouping_1_result.csv