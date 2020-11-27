# Portfolio

# [Project 1: House Prices: Advanced Regression Techniques](https://github.com/TheBigTicket02/House-Prices-Advanced)

* Predicted sales prices using tree-based and non-tree-based models
* Engineered features and applied different feature selection techniques (SHAP, L1) for appropriate models
* Optimized ElasticNet, SVR, Ridge, CatBoost, XGBoost, LightGBM to reach the best model
* Reached top 1%



| Models        | 10f-CV: RMSLE (mean)|RMSLE (std)|
| ------------- |:-------------------:|:---------:|
| SVR           | 0.0998              | 0.00852   |
| Ridge         | 0.1050              | 0.00845   |
| ElasticNet    | 0.1035              | 0.00909   |
| CatBoost      | 0.1037              | 0.00956   |
| LightGBM      | 0.1028              | 0.00948   |
| XGBoost       | 0.0993              | 0.00989   |

[Detailed notebook](https://www.kaggle.com/alexalex02/house-prices-advanced-feature-engineering)

# [Project 2: Car Model Classification](https://github.com/TheBigTicket02/Car-Classifier)

* Fine-tuned ResNet50 and EfficientNet('b5') using different training stages
* Added CLI interface
* Gained insights of best predictions with Gradient-based and Occlusion based interpretability algorithms
* Created Web Application using Streamlit
* Deployed Web App on Azure App Service using Docker

![](/images/2.1.png)

| Models        | Top 1 Accuracy|Top 2 Acc|
| ------------- |:-------------:|:-------:|
| Resnet50      | 89.14         | 95.11   |
| EffNetB5 (I)  | 93.75         | 97.82   |
| EffNetB5 (II) | 90.97         | 96.34   |

[Detailed notebook](https://www.kaggle.com/alexalex02/car-classifier-93-75-inference-web-app)
