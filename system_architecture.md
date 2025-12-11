```
Mining Process Data Sources
    |
    v
Data Ingestion & Preprocessing
    |--- Load raw plant data
    |--- Handle errors / encoding / date formats
    |--- Convert categorical & numeric fields
    |--- Remove invalid rows
    |
    v
Feature Engineering Layer
    |--- Rolling averages / lag features
    |--- Domain-specific features (e.g., % Fe)
    |--- Train / val / test split
    |
    v
Model Training & Evaluation
    |--- Baseline models (LR, RF, etc.)
    |--- XGBoost as best performer
    |--- Metrics (MAE, RMSE, R2)
    |--- Model comparison & ranking
    |
    v
Model Deployment Layer
    |--- Export trained model
    |--- Serve via API or batch prediction
    |--- Real-time or scheduled scoring
    |
    v
End User Applications
    |--- Operator dashboard
    |--- Daily reports
    |--- Alerts on abnormal predictions
```

 
## 🌐 Data Flow & System Operation

### **1. Data Sources**
Mining plant sensors measure:
- Flow rates  
- Levels  
- Pressure  
- Reagents  
- Composition metrics (e.g., % Iron Concentrate)

Data is stored either in:
- Process historians (e.g., OSIsoft PI), or  
- Periodic CSV exports (used in this project).

### **2. Data Ingestion**
The system ingests new data via scheduled ETL jobs that:
- Detect encoding automatically  
- Parse date columns  
- Convert numeric columns safely  
- Filter corrupted or impossible rows  

### **3. Feature Engineering**
To enhance predictive performance, the pipeline applies:
- Rolling/lag features  
- Domain-specific variables  
- Strong input signal: **% Iron Concentrate**, proven to significantly improve accuracy  
- Train/validation/test splitting  

### **4. Model Inference (Prediction Layer)**

#### **Real-Time Mode**
- A microservice (e.g., FastAPI) receives sensor inputs.
- Model responds with predictions (e.g., recovery, Fe grade).

#### **Batch Mode**
- A daily/hourly file is processed.
- Predictions are stored or delivered to dashboards.

### **5. Delivery to End Users**
Outputs can be delivered via:
- Web dashboards (PowerBI, Grafana)  
- Automated reports  
- Alerts when predicted values exceed defined thresholds  

Examples:
- Predicting a drop in recovery rate  
- Warning operators early about drift  
- Highlighting abnormal process conditions  

# 🚀 Next Steps and Roadmap

## **1. Improve Data Quality**
- Replace CSV ingestion with live historian API connection  
- Automated outlier detection and data smoothing  
- Build reusable preprocessing pipelines  

## **2. Enhance Model Performance**
- Apply hyperparameter tuning (Optuna)  
- Add temporal deep learning models: LSTM, TFT  
- Feature selection based on permutation importance  

## **3. Deployment at Scale**
- Containerize with Docker  
- Deploy via:
  - FastAPI endpoint  
  - Azure ML / AWS Sagemaker  
  - On-prem industrial cluster  

## **4. Monitoring & Maintenance**
- Track model drift  
- Log predictions vs. actuals  
- Schedule retraining as new data arrives  

## **5. UI & Visualization**
- Build a lightweight operator dashboard featuring:
  - Real-time predictions  
  - Feature importance visualization  
  - System health status  
- Integrate optional alerting system  
"""

output_path = "/mnt/data/system_architecture.md"

pypandoc.convert_text(
    content,
    'md',
    format='md',
    outputfile=output_path,
    extra_args=['--standalone']
)

output_path

