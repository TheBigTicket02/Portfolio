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

# [Project 3: Semantic Segmentation of High-Resolution Aerial Images](https://github.com/TheBigTicket02/Aerial-Semantic-Segmentation)

* Predicted per-pixel semantic labeling for 8 classes.

![](/images/3.1.png)

| Building | Tree | Clutter | Road | Vegetation|Static Car| Moving Car| Human| mIoU |
| ---------|:-----|:-------:|:----:|:---------:|:--------:|:---------:|:----:|:----:|
| 86.79    | 74.16| 58.01   | 71.08| 53.59     |51.34     |39.27      |22.21 |57.06 |

![](/images/3.2.png)

[Detailed notebook](https://www.kaggle.com/alexalex02/car-classifier-93-75-inference-web-app)

# [Project 4: Sentiment Analysis of Amazon Reviews - DistilBert](https://github.com/TheBigTicket02/Sentiment-Analysis-DistilBert)

* Created baseline model - Logistic Regression with Tf-Idf
* Fine-tuned DistilBert
* Applied several techniques to increase inference speed and decrease size on GPU and CPU

| Models       | Accuracy |
| -------------|:--------:|
| Log Reg      | 90.29    |
| DistilBert   | 96.22    |

[Detailed notebook](https://www.kaggle.com/alexalex02/sentiment-analysis-distilbert-amazon-reviews)

# [Project 5: NLP Transformers Inference Optimization](https://github.com/TheBigTicket02/Sentiment-Analysis-DistilBert)

* TorchScript
* Dynamic Quantization
* ONNX Runtime

### CPU

![](/images/4.1.1.png)

### GPU

![](/images/4.2.1.png)

[Detailed notebook on optimization](https://www.kaggle.com/alexalex02/nlp-transformers-inference-optimization)

# [Project 6: Deep Convolutional Generative Adversarial Network (Pytorch C++)](https://github.com/TheBigTicket02/DC-GAN)

* Trained DC-GAN to generate faces using [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset)

### After 2 epochs

![](/images/5.1.png)

### After 5 epochs

![](/images/5.2.png)

### After 25 epochs

![](/images/5.3.png)

[Detailed notebook](https://www.kaggle.com/alexalex02/dc-gan-pytorch-c)

# [Project 7: Tweet Sentiment Extraction - RoBERTa](https://github.com/TheBigTicket02/Tweet-Sentiment-Extraction-RoBERTa)

* Goal: Extract support phrases for sentiment labels

![](/images/6.1.png)

* Evaluation Metric: word-level Jaccard score

![](/images/6.2.jpg)

* Trained RoBERTa model

| Model       | Jaccard Score |
| ------------|:-------------:|
| RoBERTa     | 0.7017        |

* Build simple Web App to enter tweet and return support phrases
