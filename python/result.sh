#!/bin/bash

years_back=5
prediction_year=2028
medal_type="Bronze"
save=1
random_seed=2028

use_abundant_values=(1)
model_types=("SVM" "RandomForest" "LinearRegression" "Ridge" "Lasso" "WeightLinearRegression" "RidgeCV" "LassoCV")

for use_abundant in "${use_abundant_values[@]}"; do
  for model_type in "${model_types[@]}"; do
    echo "Running with use_abundant=${use_abundant}, model_type=${model_type}"
    python prediction.py \
      --use_abundant $use_abundant \
      --years_back $years_back \
      --prediction_year $prediction_year \
      --model_type $model_type \
      --medal_type $medal_type \
      --random_seed $random_seed \
      --save $save
  done
done

python cal_avg.py --years_back $years_back --medal_type $medal_type
