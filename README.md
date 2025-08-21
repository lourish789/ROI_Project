# JAMB Performance Prediction Model

A machine learning classification model to predict Nigerian students' JAMB performance levels based on socio-academic features. This project helps identify at-risk learners and provides insights for improving exam outcomes.

## Project Overview

**Prediction Categories:**
- **Below Average**: 0 - 199 points
- **Pass**: 200 - 400 points

**Features Used:** 15 socio-academic indicators including:
- Distance to school
- Type of school attended
- Teacher ratings
- Study habits
- And more...

**Live Demo:** [Try the model here](https://huggingface.co/spaces/Fluospark128/ROI_Project)

## Dataset

**Source:** [Kaggle - Students Performance in 2024 JAMB](https://www.kaggle.com/datasets/idowuadamo/students-performance-in-2024-jamb)

## Project Structure

```
ROI_Project/
├── JAMB_exam_scores                 # Data (CSV format)
├── JAMB_prediction.ipynb            # Jupyter notebook for EDA, model training and evaluation
├── model1.pkl               # Saved ML model (pickle file)
├── app.py               # Gradio deployment script
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/lourish788/ROI_Project.git
cd ROI_Project
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/idowuadamo/students-performance-in-2024-jamb)
2. Place the CSV file in the folder
3. Ensure the dataset contains all 15 required features

### 4. Train the Model

```bash
python JAMB_prediction.ipynb
```

This script will:
- Clean and preprocess the dataset
- Train multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, CatBoost, LGBM)
- Perform hyperparameter tuning
- Saved the best model to `model1.pkl`

### 5. Deploy Locally

```bash
python app.py
```

Open the provided local URL to test predictions interactively.

## Model Performance

- **Accuracy:** 77.7%
- **Best Algorithm:** Logistic Regression
- **Evaluation Metrics:** Precision, Recall, F1-Score, Confusion Matrix

## Technologies Used

- **Python 3.8+**
- **Machine Learning:** Scikit-learn, XGBoost, CatBoost, LightGBM
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Gradio, Hugging Face Spaces
- **Development:** Jupyter Notebook

## Requirements

Create a `requirements.txt` file with:

```txt
gradio
pandas
numpy
scikit-learn
xgboost
catboost
lightgbm 
matplotlib
seaborn

```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Impact & Applications

This model can help:
- **Teachers:** Identify at-risk students early for targeted support
- **Parents:** Understand factors affecting their children's performance
- **Policymakers:** Design data-driven educational interventions
- **NGOs:** Target resources effectively for maximum impact

## Reproduction Steps

1. **Fork/Clone** this repository
2. **Download** the dataset from the provided Kaggle link
3. **Set up** your Python environment with the required packages
4. **Run** the training script to reproduce results
5. **Deploy** using Gradio for interactive testing

