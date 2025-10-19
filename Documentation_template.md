# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** code ninjas
**Team Members:** Somya Upadhyay, Sagar Agarwal, Aviral Sharma, Tathagata Sen
**Submission Date:** October 13, 2025

---

## 1. Executive Summary

Our solution implements a multimodal ensemble approach combining advanced text processing with deep image features to predict product prices. We developed a comprehensive pipeline that extracts semantic features from product descriptions using TF-IDF and ResNet50-based image embeddings, achieving competitive performance through ensemble modeling and SMAPE-optimized calibration techniques.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

We interpreted the pricing challenge as a multimodal regression problem where both textual product descriptions and visual product images contain complementary pricing signals. Our exploratory data analysis revealed several key insights about the relationship between product attributes and pricing.

**Key Observations:**
- Product descriptions contain rich information about Item Pack Quantity (IPQ), brand indicators, and quality descriptors
- Image features provide visual cues about product size, materials, and premium characteristics  
- Price distribution spans a wide range ($0.54 - $672.20) requiring robust scaling and calibration
- Text length, keyword presence, and numerical values in descriptions correlate with price ranges
- Systematic underestimation was a major challenge requiring specialized calibration techniques

### 2.2 Solution Strategy

Our approach combines multiple data modalities through an ensemble framework optimized specifically for SMAPE minimization.

**Approach Type:** Multimodal Ensemble  
**Core Innovation:** SMAPE-optimized ensemble calibration with ResNet50 image features and advanced text processing

---

## 3. Model Architecture

### 3.1 Architecture Overview

Our pipeline consists of three main components:
1. **Text Feature Extraction**: TF-IDF vectorization with SVD dimensionality reduction
2. **Image Feature Extraction**: ResNet50 CNN with PCA compression  
3. **Ensemble Modeling**: Random Forest, Gradient Boosting, and Neural Network combination

### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: Enhanced IPQ extraction, text statistics, keyword detection
- [x] Model type: TF-IDF with n-grams (1,2) + TruncatedSVD
- [x] Key parameters: max_features=1000, n_components=100, min_df=2

**Image Processing Pipeline:**
- [x] Preprocessing steps: Resize (224x224), normalization, tensor conversion
- [x] Model type: Pre-trained ResNet50 + PCA dimensionality reduction
- [x] Key parameters: 2048 → 77 dimensions, batch_size=100

**Ensemble Configuration:**
- Random Forest: n_estimators=500, max_depth=25, max_features='sqrt'
- Gradient Boosting: n_estimators=500, max_depth=12, learning_rate=0.03
- Neural Network: hidden_layers=(200,100,50), activation='relu', alpha=0.001

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 29.8 (optimized ensemble on 100-sample validation set)
- **Other Metrics:** MAE: $16.52, RMSE: $34.92
- **Feature Count:** 177 total features (100 text + 77 image)
- **Training Samples:** 40000 samples (optimized for image processing efficiency)

### 4.2 Performance Analysis
- Successfully achieved target SMAPE < 30 through ensemble calibration
- Image features contributed 10-15% performance improvement over text-only baseline
- Advanced text processing (TF-IDF + enhanced IPQ extraction) provided 15-25% improvement
- Ensemble approach reduced overfitting and improved generalization

## 5. Conclusion

Our multimodal ensemble approach successfully combines text and image features to achieve competitive pricing predictions with SMAPE < 30. The key innovations include ResNet50-based image feature extraction, enhanced text processing with multiple IPQ patterns, and SMAPE-optimized ensemble calibration. The solution demonstrates the value of combining multiple data modalities and specialized optimization techniques for regression tasks with asymmetric error metrics.

---

## Appendix

### A. Code Artifacts
**Main Implementation Files:**
- `fresh_train_75k_smape_under_30.py` - Complete training pipeline
- `image_feature_extractor.py` - ResNet50 image processing
- `config.py` - Configuration parameters
- `src/utils.py` - Image download utilities

### B. Additional Results
**Model Comparison:**
- Text-only baseline: SMAPE ~40-50
- Text + Image features: SMAPE ~35-40  
- Optimized ensemble: SMAPE 29.8

**Feature Engineering:**
- 100 TF-IDF text features (reduced from 1000 via SVD)
- 77 image features (reduced from 2048 via PCA)
- Enhanced IPQ extraction with multiple regex patterns
- Price-indicative keyword detection (premium, budget, quality indicators)

**Technical Specifications:**
- Processing time: ~20 minutes for 1500 images
- Memory efficient batch processing for image downloads
- Robust error handling for failed image downloads
- Cross-validated hyperparameter optimization

---

**Note:** This solution prioritizes practical implementation with efficient image processing while maintaining competitive performance. The approach balances computational efficiency with model accuracy, making it suitable for production deployment.