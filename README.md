# project_deep: repository for my project in distance based confidence for regression

The project is split into two parts: 
1.	Reimplement the method described in (Mandelbaum & Weinshall, 2017) and reproduce their results.
2.	Expand the distance-based metric for regression problems and compare its performance against MC dropout for on 3 datasets, the main criteria for comparison is selective risk described in (Geifman, Uziel, & El-Yaniv, 2019)

for part one use train_distance_based.py for training the models, and evaluate_distance_based.py for evaluation 

##  results for part 1. are:
values are auc on binary classification of the conf. score on weather the model predicted the right label or not, more detailes in the paper

### on CIFAR-100 dataset
| Conf. score | Reg. (acc 61.67%) | Dist. (acc 64.48%) | Dist. Squared (acc 63.89%) |
|-------------|-------------------|--------------------|----------------------------|
| Max Margin  | 0.83331           | 0.8394             | 0.8473                     |
| Entropy     | 0.8382            | 0.8391             | 0.8442                     |
| Distance    | 0.8534            | **0.8559**            | **0.8626**                     |


### on CIFAR-10 dataset
| Conf. score | Reg. (acc=90.34%) | Dist. (acc 90.75%) |
|-------------|-------------------|--------------------|
| Margin      | 0.9045            | 0.8567             |
| Entropy     | **0.9052**            | 0.8557             |
| Distance    | 0.8707            | 0.8540             |


### on SVHN dataset
| Conf. score | Reg. (acc 92.19%) | Dist. (acc 92.08%) |
|-------------|-------------------|--------------------|
| Margin      | 0.9200            | **0.9214**             |
| Entropy     | 0.9145            | 0.9178             |
| Distance    | 0.8993            |  0.9113            |
--------------------------------------------------------------------------------------------

for part two use train_distance_based_regression.py for training the models, and risk_evaluation.py for evaluation 
## results for part 2. are:
the values are selective risk as described in (Geifman, Uziel, & El-Yaniv, 2019)
improvement is first column relative to second column

### on facial keypoints dataset

comparing min dist. with different training methods

| coverage | risk (RMSE) trained loss: MSE       | risk (RMSE) MSE + distance    | Improvement % |
|----------|-------------------|----------------|---------------|
|          |                   |                |               |
| 0.5      | 1.397551          | 1.400984       | -0.245643987  |
| 0.6      | 1.435375          | 1.457736       | -1.557850736  |
| 0.7      | 1.4601            | 1.496928       | -2.522292994  |
| 0.8      | 1.53422           | 1.538056       | -0.250029331  |
| 0.85     | 1.569422          | 1.566886       | 0.161588152   |
| 0.9      | 1.60538           | 1.622597       | -1.072456365  |
| 0.95     | 1.640438          | 1.647965       | -0.458840871  |
| 1        | 1.700737          | 1.718866       | -1.065949644  |

comparing min dist. vs MC dropout

| coverage | risk (RMSE) min dist. | risk (RMSE) MC dropout | Improvement % |
|----------|-------------|-------------|---------------|
|          |             |             |               |
| 0.5      | 1.397551    | 1.59643775  | 12.4581588    |
| 0.6      | 1.435375    | 1.614019    | 11.06827119   |
| 0.7      | 1.4601      | 1.6144965   | 9.563136247   |
| 0.8      | 1.53422     | 1.6299505   | 5.873215168   |
| 0.85     | 1.569422    | 1.639529    | 4.276045132   |
| 0.9      | 1.60538     | 1.658925    | 3.227692632   |
| 0.95     | 1.640438    | 1.67232     | 1.90645331    |
| 1        | 1.700737    | 1.73235825  | 1.82532972    |

### on concrete strength dataset
comparing min dist. with different training methods

| coverage | risk (MSE) trained loss: MSE         | risk (MSE) MSE + distance    | Improvement % |
|----------|-------------------|----------------|---------------|
|          |                   |                |               |
| 0.5      | 22.452653         | 23.058849      | -2.699885844  |
| 0.6      | 21.273017         | 23.705395      | -11.43409983  |
| 0.7      | 22.727649         | 24.112884      | -6.094933092  |
| 0.8      | 23.31525          | 25.329021      | -8.637140927  |
| 0.85     | 23.858668         | 25.695215      | -7.697609104  |
| 0.9      | 27.584914         | 31.63828       | -14.69414043  |
| 0.95     | 30.152303         | 31.537648      | -4.594491505  |
| 1        | 32.749712         | 32.647679      | 0.311553885   |

comparing min dist. vs MC dropout

| coverage          | risk (MSE) min dist. | risk (MSE) MC dropout | Improvement % |
|-------------------|------------|------------|---------------|
|                   |            |            |               |
| 0.5               | 22.452653  | 19.3217596 | -16.203976    |
| 0.6               | 21.273017  | 19.8097438 | -7.3866336    |
| 0.7               | 22.727649  | 25.2252716 | 9.90127139    |
| 0.8               | 23.31525   | 31.2687252 | 25.4358793    |
| 0.85              | 23.858668  | 33.3958988 | 28.5580898    |
| 0.9               | 27.584914  | 33.4714252 | 17.5866762    |
| 0.95              | 30.152303  | 33.2364412 | 9.27938759    |
| 1                 | 32.749712  | 32.4167376 | -1.0271681    |
| Approximate AURCC | 16.1513663 | 15.0103195 | 7.06470758    |


for complete analysis and report please read the project [summary]( https://technionmail-my.sharepoint.com/:w:/g/personal/mdabbah_campus_technion_ac_il/EeWVB2q-jSdLjiDXN0vsf98BpZG-j-QgjyDRfNrXNWwRuA?e=TqC9dC)
