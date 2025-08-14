# Openlearn-Capstone-Project
Machine Learning analysis and persona segmentation for mental health in tech 
# 🧠 Mental Wellness Analysis & Support Strategy  
**OpenLearn Cohort 1.0 – Capstone Project 2025**

## 📌 Project Overview
This project analyzes the **Mental Health in Tech Survey** dataset (OSMI) to understand factors influencing mental health among tech employees.  
It uses **classification, regression, and clustering** to uncover insights, predict treatment-seeking behavior, and identify key employee personas.

An **interactive Streamlit web app** is deployed to allow HR teams, researchers, and policymakers to explore findings, run predictions, and visualize mental health trends in the tech workplace.

---

## 🎯 Objectives
- **Classification** – Predict whether an employee is likely to seek mental health treatment.
- **Regression** – Predict respondent’s age based on workplace and personal factors.
- **Clustering** – Segment employees into mental health personas for targeted HR policies.
- **EDA** – Explore trends, correlations, and patterns in demographics, workplace culture, and mental health attitudes.

---

## 📊 Dataset
- **Source:** [Mental Health in Tech Survey – OSMI](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Size:** 1,500+ responses
- **Key Features:**
  - Demographics (Age, Gender, Country)
  - Workplace Environment (Benefits, Leave Policies, Remote Work)
  - Mental Health History (Personal & Family)
  - Attitudes towards mental health

---

## 🛠️ Project Workflow
1. **Data Cleaning & Preprocessing**
   - Handle missing values, anomalies, and outliers
   - Encode categorical features
2. **Exploratory Data Analysis (EDA)**
   - Univariate, Bivariate & Multivariate analysis
   - Correlation heatmaps and insights
3. **Machine Learning Models**
   - **Classification:** Logistic Regression, Random Forest, XGBoost, SVM
   - **Regression:** Linear Regression, XGBoost Regressor
   - **Clustering:** KMeans, Agglomerative Clustering, DBSCAN
4. **Model Evaluation**
   - Accuracy, ROC-AUC, Confusion Matrix, F1 Score
   - RMSE, MAE, R² for regression
   - Silhouette Score for clustering
5. **Streamlit Deployment**
   - Interactive EDA visualizations
   - Treatment & age prediction forms
   - Cluster visualization & persona descriptions

---

---
## 📂 Repository Structure
```text
├── Images/                            
│   ├── subplots1.png
│   ├── subplots2.png
│   ├── subplots3.png
│   ├── subplots4.png
│   ├── subplots5.png
│   ├── subplots6.png
│   ├── subplots7.png
│   ├── subplots8.png
│   ├── subplots9.png
│   ├── subplots10.png
│   ├── knn.png
│   ├── auc.png
│   ├── knnr.png
│   ├── pca.png
│   ├── tsne.png
│   ├── agg.png
│   ├── kmeans.png
│   ├── kmeanselb.png
│   └── aggcluster.png

├── Pickle Files/                 
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   └── df.pkl                
       
├── Notebooks/                       
│   ├── eda-capstone-m.ipynb
│   ├── supervised-learning-task-classification-m.ipynb
│   ├── supervised-learning-regression-m.ipynb
│   └── unsupervised-learning-m.ipynb

├── app.py                            
├── survey.csv                        
└── README.md

```

---
## 🔗 Important Links
[**Click here to view the Streamlit app**](https://openlearn-capstone-project-ffqxpxbc6h94a4fhhaz6uu.streamlit.app/)
- **Technical Report:** [Link to Medium Blog]([https://medium.com/@](https://medium.com/@mitanshihs.ec.24/mental-heath-in-tech-machine-learning-analysis-persona-segmentation-e45628493737)) 

---

## Acknowledgements
- OpenLearn Cohort 1.0 Pathfinders & fellow Pioneers
