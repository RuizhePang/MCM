use_abundant=1
years_back=3
prediction_year=2024

model_type='SVM'
#model_type='RandomForest'
#model_type='DicisionTree'
#model_type='LinearRegression'

python prediction.py \
    --use_abundant $use_abundant \
    --years_back $years_back \
    --prediction_year $prediction_year \
    --model_type $model_type
