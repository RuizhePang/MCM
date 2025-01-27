use_abundant=1
years_back=5
prediction_year=2028
save=0
random_seed=2

medal_type='Total'
#medal_type='Gold'
#medal_type='Silver'
#medal_type='Bronze'

#model_type='SVM'
#model_type='RandomForest'
#model_type='DecisionTree'
#model_type='LinearRegression'
#model_type='Ridge'
#model_type='Lasso'
#model_type='WeightLinearRegression'
#model_type='RidgeCV'
model_type='LassoCV'

python prediction.py \
    --use_abundant $use_abundant \
    --years_back $years_back \
    --prediction_year $prediction_year \
    --model_type $model_type \
    --medal_type $medal_type \
    --save $save \
    --random_seed $random_seed
