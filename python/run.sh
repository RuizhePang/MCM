use_abundant=1
years_back=5
prediction_year=2028

medal_type='Total'
#medal_type='Gold'
#medal_type='Silver'
#medal_type='Bronze'

#model_type='SVM'
#model_type='RandomForest'
#model_type='DicisionTree'
model_type='LinearRegression'

python prediction.py \
    --use_abundant $use_abundant \
    --years_back $years_back \
    --prediction_year $prediction_year \
    --model_type $model_type \
    --medal_type $medal_type
