# ROI_Project; JAMB Performance Prediction Model

Team ROI built a machine learning classification model to predict Nigerian students’ JAMB performance level; Below Average(0 - 199) , or Pass(200 - 400) based on 15 socio-academic features such as distance to school, type of school, teacher ratings, and study habits. 

The goal is to provide insights that help students, teachers, and policymakers identify at-risk learners and improve exam outcomes.

Table of Contents 
Project Structure 

Dataset source: https://www.kaggle.com/datasets/idowuadamo/students-performance-in-2024-jamb

The more was built using the following framework;
Python
Jupyter Notebook 
Gradio 
Huggingface 

We performed exploratory data analysis on the dataset to extract insights and check the relationship between the variables 





To deploy the model, the model was saved to a pickle file file 

Then, it was deployed via gradio on huggingfaces. here is the link tonthe web app; https://huggingface.co/spaces/Fluospark128/ROI_Project















---

##📂 Project Structure

├── data/                 # Dataset files (CSV format)
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── models/               # Saved ML models (pickle files)
├── jamb_prediction.py    # Training and evaluation script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation


---

##⚙️ Features

Multiple ML algorithms compared (Logistic Regression, Decision Trees, Random Forest, XGBoost, CatBoost, LGBM.)

Hyperparameter tuning #with GridSearchCV / RandomizedSearchCV

#Cross-validation for robust evaluation

Metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

Exported best model in .pkl format

Deployment-ready with Gradio or Streamlit



---

##🚀 Setup Instructions

1. Clone the Repository

git clone https://github.com/lourish788/ROI_Project.git
cd jamb-prediction

2. Create Virtual Environment & Install Dependencies

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

3. Prepare Dataset

Place your dataset file (e.g. jamb_exam_results.csv) inside the data/ folder.

Ensure the dataset has the required 15 features.


4. Train the Model

python jamb_prediction.py

This will:

Clean and preprocess the dataset

Train multiple models (Logistic Regression, Random Forest, CatBoost, etc.)

Run hyperparameter tuning and cross-validation

Save the best model into models/best_model.pkl


5. Evaluate the Model

Evaluation metrics will be displayed in the terminal and saved to a results file.

6. Deploy with Gradio

python app.py

Then open the local link to test predictions interactively.



---

##📊 Example Output

Accuracy: 77.7%

Best Model: Logistic Regression

Confusion Matrix: Shows correct/incorrect classifications across 3 performance categories



---

##🌍 Impact

This model can:

Help teachers identify at-risk students early

Guide parents in supporting children

Assist NGOs and policymakers in targeting interventions



---

##📌 Requirements

Python 3.8+

Pandas, NumPy, Scikit-learn, CatBoost, XGBoost

Matplotlib, Seaborn (for visualization)

Gradio or Streamlit (for deployment)


Install them via:

pip install -r requirements.txt


---

🤝 Contribution

Feel free to fork this repo and submit PRs to improve the model, add datasets, or extend deployment options.
