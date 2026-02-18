# Karachi Air Quality Intelligence (AQI) System

> **Advanced Machine Learning Pipeline for Real-Time Air Quality Forecasting**

> *Developed by Muhammad Ehsan | Production-Grade Architecture | Automated CI/CD*

**[View Live Dashboard](https://10pearls-aqi.streamlit.app/)**

---

## Project Overview

The **Karachi AQI Prediction System** is an end-to-end, automated machine learning platform designed to forecast the Air Quality Index (AQI) for Karachi, Pakistan, for the next 72 hours.

Unlike static analysis tools, this system features a **self-correcting Machine Learning pipeline** that automatically fetches new data, retrains multiple algorithms (Linear Regression, XGBoost, Random Forest), and dynamically promotes the best-performing model to production without human intervention. The system is built on a modular, scalable architecture suitable for cloud deployment.

## Key Features

* **Adaptive Machine Learning**: The system trains an ensemble of models daily and automatically selects the one with the lowest Root Mean Square Error (RMSE) for inference.
* **Fully Automated Pipeline**: GitHub Actions manage hourly data ingestion and daily model retraining, ensuring the forecast is always based on the latest atmospheric conditions.
* **Modular Production Architecture**: The codebase follows strict separation of concerns, dividing the application into frontend, backend logic, and model artifacts for maintainability.
* **Cloud-Native Data Store**: Utilizes MongoDB Atlas as a scalable Feature Store to manage historical weather and pollution data.
* **Interactive Analytics Dashboard**: A professional Streamlit interface providing real-time gauges, 72-hour trend analysis, and model performance metrics.

---

## Project Structure

The repository is organized into distinct modules to ensure scalability and ease of navigation:

```text
karachi-aqi-intelligence/
├── app/                    # Frontend Application
│   └── dashboard.py        # Main Streamlit Dashboard Interface
│
├── src/                    # Backend Logic & Pipelines
│   ├── data_ingestion.py   # API Connectivity & Data Fetching
│   ├── database.py         # MongoDB Connection Manager
│   ├── preprocessing.py    # Data Cleaning & Outlier Removal
│   ├── feature_engineering.py # Lag Generation & Rolling Averages
│   └── modeling.py         # Model Training & Inference Logic
│
├── models/                 # Model Artifacts
│   ├── model.pkl           # The active, best-performing model
│   └── features.pkl        # Serialized feature list for consistency
│
├── docs/                   # Documentation & Reports
│   └── Project_Report.pdf  # Technical analysis
│
├── notebooks/              # Research & Experiments
│   └── EDA.ipynb
│   └── Shap_analysis.ipynb
│
├── .github/workflows/      # CI/CD Automation
│   ├── daily_retrain.yml   # Scheduled Model Retraining
│   └── hourly_data_update.yml # Scheduled Data Fetching
│
├── Dockerfile              # Containerization Configuration
├── requirements.txt        # Python Dependencies
└── .env                    # Environment Variables (Excluded from Git)

```

---

## Model Performance

The system evaluates models dynamically during every training cycle. Below are the performance metrics from the latest deployment:

| Model Type | RMSE (Error) | R² Score | Status |
| --- | --- | --- | --- |
| **Linear Regression** | **3.10** | **0.96** | **Active** |
| XGBoost | 3.56 | 0.95 | Candidate |
| Random Forest | 3.78 | 0.95 | Candidate |

*Note: Lower RMSE indicates higher prediction accuracy. The system automatically switches the active model if a candidate outperforms the current production model.*

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/MuhammadEhsan02/10Pearls-AQI.git
cd 10Pearls-AQI

```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt

```

### 3. Configure Environment

Create a `.env` file in the root directory and add your MongoDB credentials:

```env
MONGO_URI=your_mongodb_connection_string
DB_NAME=aqi_db

```

### 4. Run the Pipeline (Optional)

To manually trigger the data pipeline and model training:

```bash
# Fetch the latest data
python src/data_ingestion.py

# Train and save the model
python src/modeling.py

```

### 5. Launch the Dashboard

Start the local development server:

```bash
streamlit run app/dashboard.py

```

---

## License

This project is open-source and available under the MIT License.
