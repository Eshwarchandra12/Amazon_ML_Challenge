Product Price Prediction using NLP + LightGBM

Overview:

> This project demonstrates how traditional machine learning techniques combined with NLP can be applied to solve real-world problems such as price estimation, catalog optimization, and product analytics.
> Accurately estimating product prices from unstructured textual descriptions is a challenging real-world problem due to inconsistencies in formatting and missing structured attributes.
> This project builds an end-to-end machine learning pipeline using NLP techniques to extract meaningful information from product catalog data and predict prices effectively.
> Text data is cleaned, transformed using TF-IDF, reduced via SVD, and modeled using LightGBM with cross-validation for robust predictions. The system demonstrates how traditional ML + NLP can solve real-world pricing problems.

Dataset
- Train: 75,000 samples (text + price)
- Test: 75,000 samples (text only)
- Features: catalog_content, image_link (unused), price

Approach
1. Text Preprocessing: cleaning, normalization, regex filtering
2. Feature Engineering: TF-IDF (50K features, ngrams)
3. Dimensionality Reduction: SVD (256 components)
4. Modeling: LightGBM with 5-fold CV and log transformation

Tech Stack
Python, Pandas, NumPy, Scikit-learn, LightGBM, TQDM

Results
Validation MAE: 12.68
SMAPE: 55.56%

Key Highlights
- End-to-end NLP regression pipeline
- Dimensionality reduction using SVD
- Cross-validation for robustness
- Efficient gradient boosting model
Limitations
- No image features used
- Baseline TF-IDF approach
- Limited feature engineering
Future Improvements
- Use BERT/Sentence-BERT embeddings
- Multimodal learning (text + image)
- Hyperparameter tuning (Optuna)
- Deploy via FastAPI
