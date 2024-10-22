# Fraud Detection Using Deep Learning

This project demonstrates the development and implementation of a deep learning model to detect fraudulent transactions in a highly imbalanced credit card transaction dataset. The model was built using a Feedforward Neural Network (FNN) and leverages advanced machine learning techniques to improve fraud detection rates while maintaining operational efficiency.

## Project Overview

- **Project Name**: Fraud Detection Using Deep Learning Models
- **Author**: Ali Wahaj

## Colab Notebook

You can access the complete code for the project in the following Google Colab Notebook:  
[CA_1_Deep_Learning_FNL.ipynb](https://colab.research.google.com/drive/1YHoWzdczMpYnQYzVteV7CW5t9j0SO4_M)

## Project Objective

The main objective of this project is to build and evaluate a deep learning model that can effectively detect fraudulent transactions in a large-scale credit card dataset. The focus is on developing a model with high accuracy, precision, and recall while maintaining a low false-positive rate to minimize disruptions in legitimate transactions.

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Number of Transactions**: 284,807
- **Fraudulent Transactions**: 492
- **Non-fraudulent Transactions**: 284,315

The dataset is highly imbalanced, with fraudulent transactions representing only about 0.17% of the total transactions.

## Methodology

The project followed the CRISP-DM methodology, which includes the following steps:

1. **Business Understanding**:
   - The main goal is to detect fraudulent transactions with minimal false positives.
   - The model aims to optimize precision and recall, as well as minimize operational costs due to false positives.

2. **Data Understanding**:
   - The dataset contains anonymized transaction data, with 28 features derived from Principal Component Analysis (PCA) to protect privacy.
   - The dataset is highly imbalanced, which posed a challenge for effective fraud detection.

3. **Data Preprocessing**:
   - Used StandardScaler for feature scaling.
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance, balancing the fraudulent and non-fraudulent classes.

4. **Modeling**:
   - Developed a Feedforward Neural Network (FNN) with multiple dense layers, followed by Batch Normalization and Dropout to prevent overfitting.
   - Trained the model using the Adam optimizer and binary cross-entropy loss function.

5. **Evaluation**:
   - The model was evaluated based on accuracy, precision, recall, F1 score, and the Area Under the ROC Curve (AUC).
   - Achieved a validation accuracy of 99.76% and an AUC score of 0.91.

## Model Architecture

- **Input Layer**: 28 input features from the dataset.
- **Hidden Layers**: Multiple fully connected (dense) layers with Batch Normalization and Dropout.
- **Output Layer**: A single output node for binary classification (fraud/non-fraud).
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Training**: The model was trained over 30 epochs, with early stopping applied to prevent overfitting.

## Results

- **Validation Accuracy**: 99.76%
- **AUC Score**: 0.91
- **F1 Score for Fraudulent Class**: 0.82
- **Confusion Matrix**: Minimal false positives and false negatives, demonstrating the model's efficiency in identifying fraudulent transactions without causing unnecessary disruptions for legitimate users.

## Visualizations

1. **Correlation Matrix**: Shows relationships among the dataset’s features.
2. **Class Distribution Plot**: Highlights the significant class imbalance.
3. **Training vs Validation Accuracy**: Demonstrates the model's learning progress and ability to generalize.
4. **Precision-Recall Curve**: Illustrates the tradeoff between precision and recall, with a good balance at a threshold of 0.97.

## Economic Impact

A cost-benefit analysis was performed to assess the financial implications of deploying the model:

- **Total Cost due to Errors**: $29,450
- **Potential Savings from Detected Fraud**: $28,500
- **ROI**: 90%, demonstrating the model’s economic viability.

## Future Work

1. **Improve False Negative Detection**: Enhance the model to further reduce false negatives, as undetected fraud represents a direct financial risk.
2. **Model Retraining**: Regular retraining with new data to adapt to evolving fraud patterns.
3. **Integration**: Deploy the model into real-time systems for transaction monitoring and detection.

## Deployment

The model is designed for deployment into existing fraud detection systems, with plans for real-time transaction monitoring and regular model updates based on new data.

## Conclusion

The project successfully demonstrates the application of deep learning models for fraud detection. The model achieved a high level of accuracy and demonstrated economic benefits, making it a viable solution for financial institutions looking to mitigate fraud-related risks.

---

## How to Run the Project

1. Download the notebook file `CA_1_Deep_Learning_FNL.ipynb` or access it directly via the [Google Colab Link](https://colab.research.google.com/drive/1YHoWzdczMpYnQYzVteV7CW5t9j0SO4_M).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt



### Key Elements Included:
- **Project Overview**: Brief summary of the project’s objectives and methodology.
- **Dataset and Methods**: Information about the dataset, preprocessing steps, and modeling approach.
- **Model Architecture and Results**: Details on the deep learning model and its performance metrics.
- **Visualizations**: Mention of key visual outputs like the correlation matrix and precision-recall curve.
- **Economic Impact**: A summary of the cost-benefit analysis results.
- **Deployment & Future Work**: Information about potential improvements and deployment strategies.
- **Contact Information**: A section for employers to reach out to you.

This `README.md` will give potential employers a clear understanding of the project, its significance, and your role in its development.
