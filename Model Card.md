# **Model Card: LSTM Regression Model for Flotation Plant Silica Prediction**

## **Model Details**

**Model Name:**  
LSTM-SilicaPredictor

**Model Version:**  
v1.0 (Initial Release)

**Model Type:**  
Time-series regression model

**Developers:**  
Jack Gu (Jackie)

**Release Date:**  
2025-12-10

---

## **Intended Use**

**Primary Use:**  
Predict **% Silica Concentrate** in a flotation plant using sequential process sensor data in below dataset
https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process
This model was developed to explore LSTM performance on noisy industrial datasets.

**Intended Users:**  
- Researchers in industrial machine learning  
- Anyone experimenting with LSTM-based regression

**Out-of-Scope Use Cases:**  
- Real-time industrial control or safety-critical decision systems  
- Financial, regulatory, or autonomous process decision-making  
- Use on unrelated datasets without appropriate preprocessing  
- Any other uses other than stated in Primary use

---

## **Model/Data Description**

### **Data Used**
The model is trained on the **Mining Process Flotation Plant Database**, consisting of time-series sensor readings and process assay data.

**Characteristics:**
- Time-ordered process data  
- Includes duplicated and irregular timestamps  
- Contains metallurgical features (flows, percentages, chemical indicators)  
- Target variable: **% Silica Concentrate**  

**Preprocessing Steps:**
- Multiple encoding fallback for loading the CSV  
- Conversion of comma-decimal values to standard decimals  
- Numeric conversion for all columns  
- Timestamp parsing and creation of unique ordered timestamps  
- Feature scaling using `StandardScaler`  
- Creation of two datasets: with iron and without iron

**Potential Biases:**
- Industrial operation–dependent bias (shift changes, process instability)  
- No human-related bias present  
- Possible distribution drift due to varying plant conditions

---

### **Features**
**Inputs include numerical variables such as:**
- Flow rates  
- Chemical process indicators  
- Concentrate stream variables  
- (Optional) % Iron Concentrate

**Target Variable:**
- **% Silica Concentrate**

---

### **Model Architecture**
- **Type:** LSTM neural network  
- **Layers:**  
  - 2-layer LSTM  
  - Hidden size: 128  
  - Dropout: 0.2  
- **Sequence Length:** 32  
- **Output Layer:** Fully connected linear layer  
- **Optimizer:** Adam (lr = 0.001)  
- **Loss Function:** MSE  
- Automatically uses CPU or CUDA depending on availability

---

## **Training and Evaluation**

### **Training Procedure**
- Train/validation split: **chronological 80/20**  
- Sliding window dataset creation  
- Batch size: 64  
- Epochs: 20 with early stopping (patience = 5)  
- Input and output scaling performed using `StandardScaler`  
- Best model selected based on validation loss

### **Evaluation Metrics**
- **MAE** (Mean Absolute Error)  
- **MSE** (Mean Squared Error)  
- **RMSE**  
- **R² Score** 

### **Baseline Comparison**
- Compared with classical models such as **XGBoost**  
- **XGBoost outperformed LSTM** in both speed and accuracy  
- Including **% Iron Concentrate** improved results across all models  
- All models had **negative R²**, implying dataset complexity/noise is a major issue

---

## **Ethical Considerations**

### **Fairness and Bias**
- Dataset has no demographic/human data → no social bias concerns  
- Potential bias from operational shifts or unrecorded plant conditions  
- Model may overfit to certain process states

### **Privacy**
- Dataset includes no personal information  
- No privacy risks identified

### **Security**
- No security vulnerabilities inherent to the model  
- Should not be used for autonomous control of physical equipment  

---

## **Limitations and Recommendations**

### **Known Limitations**
- Negative R² indicates weak predictive performance  
- LSTM struggles with irregular timestamps and noisy industrial data  
- Limited sequence context may miss long-term dependencies  
- Not suitable for production environments  

### **Recommendations for Use**
- Intended for **education, experimentation, and research only**  
- Consider advanced architectures for improved performance:  
  - Attention mechanisms  
  - Temporal Convolutional Networks (TCN)  
  - Transformer-based regressors  
- Perform feature engineering or smoothing to reduce noise  
- Prefer XGBoost for fast, more reliable baseline results

---

## **Additional Information**

### **References**
- Mining Process Flotation Plant Dataset  
- PyTorch Documentation  
- Scikit-learn Documentation  

### **License**
MIT License
### **Contact Information**
Developer: **Jack Gu (Jackie)**  
University of Alberta  

