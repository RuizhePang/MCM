#!/bin/bash

years_back=5
prediction_year=2028
medal_type="Total"
save=1

use_abundant_values=(0 1)
model_types=("SVM" "RandomForest" "DecisionTree" "LinearRegression")

for use_abundant in "${use_abundant_values[@]}"; do
  for model_type in "${model_types[@]}"; do
    echo "Running with use_abundant=${use_abundant}, model_type=${model_type}"
    python prediction.py \
      --use_abundant $use_abundant \
      --years_back $years_back \
      --prediction_year $prediction_year \
      --model_type $model_type \
      --medal_type $medal_type \
      --save $save
  done
done

#python cal_avg.py --years_back $years_back --medal_type $medal_type
