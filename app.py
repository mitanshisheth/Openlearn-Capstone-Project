import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, classification_report, mean_absolute_error, mean_squared_error, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost
import joblib

df = joblib.load(r"Pickle File/data.pkl")

clf_model = joblib.load(r"C:\\Users\\mitan\\Downloads\\classification_model.pkl")

reg_model = joblib.load(r"C:\\Users\\mitan\\Downloads\\regression_model.pkl")
import streamlit as st

# Custom pastel pink background
page_bg = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #FFD1DC; /* Pastel Pink */
    }
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

#layout
st.set_page_config(page_title="Mental Health Survey App", layout="wide")

st.sidebar.title("Navigation")
def footer():
    st.markdown("---")
    st.markdown("""
    <small> Model made and deployed by Mitanshi Sheth 💖🎀| 
    [LinkedIn]() • 
    [GitHub]() • 
    [X]()</small>
    """, unsafe_allow_html=True)

def Home():
    st.title("📊 OpenLearn Capstone Project ")
    st.divider()
    st.header("🌐Dataset Overview")
    st.markdown("""
           Data originates from the OSMI “Mental Health in Tech” surveys conducted in 2014 and 2016, available on Kaggle for public use.
           dataset-- [Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)       
           Topics Covered: Demographics, mental health background, workplace context, stigma and discussion comfort, support programs.
           Use Cases: Data exploration, visualization, statistical tests, machine learning models (e.g., predicting comfort in discussing mental health).      
                
                """)
    st.header("🔍Objectives for Capstone Project")
    st.markdown(""" 
                To understand the key factors influencing mental health issues among employees in the tech industry and
                build data-driven solutions for:
                Classification Task: Predict whether an individual is likely to seek mental health treatment.
                Regression Task: Predict the age of an individual based on personal and workplace attributes,
                supporting age-targeted intervention design.
                Unsupervised Task: Segment tech employees into distinct clusters based on mental health indicators
                to aid in tailored HR policies.
                """)
    st.header("Project Components🪜")
    st.markdown(""" 
            ### Part 1: Exploratory Data Analysis (EDA)
            ### Part 2: Supervised Learning Tasks : A. Classification Task. B. Regression Task
            ### Part 3: Unsupervised Learning Task
            ### Part 4: Streamlit Deployment
                """)
    footer()

def EDA():
    st.title("👩‍🔬Exploratory Data Analysis")
    st.divider()


    st.write("Raw overview of dataset with insights:")
    st.write("Total number of values: `1259`")
    st.write("Total number of features: `27`")
    st.write("Features having NaN values: \n")
    st.write("\t`state`: 451")
    st.write("\t`self_employed`: 18")
    st.write("\t`work_interference`: 264")
    st.write("\t`comments`: 1095")

    st.divider()
    st.write("### Dataset Preview:")
    st.dataframe(df.head())
    st.divider()
    st.subheader("📅Dataset columns (unused) and dtypes")
    col1, col2 = st.columns(2)
    with col1:
       st.subheader("Columns not used in any task:")
       cols_rem=['Timestamp','comments','leave','Country','state']
       for col in cols_rem:
            st.write(col)
    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)

    st.divider()

    st.header("📝Univariate Analysis")
    st.image("C:\\Users\\mitan\\Downloads\\subplots1.png", caption="Age And Gender", use_container_width=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ###
        - Mean Age of Respondents: `32`
        """)

    with col2:
        st.markdown("""
        ### 
        - Male: `78.92%`
        - Female: `19.79%`
        - Other: `1.29%`
        """) 
    st.image("C:\\Users\\mitan\\Downloads\\subplots2.png", caption="Employee distribution and Country comparision", use_container_width=True)
    st.divider()
    st.image("C:\\Users\\mitan\\Downloads\\subplots3.png", caption="More info about respondent", use_container_width=True)
    st.divider()
    st.image("C:\\Users\\mitan\\Downloads\\subplots4.png", caption="Views on mental and physical health", use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        • There are mixed opinions about bringing up physical health
        """)
    with col2:
        st.markdown("""
        • A large percentage of people would not prefer bringing Mental health issues up in meetings
        """)
    st.divider()
    st.image("C:\\Users\\mitan\\Downloads\\subplots5.png", use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        •  A lot of them would be able to communicate with a few trusted coworkers
        """)
    with col2:
        st.markdown("""
        • It is observed that many feel comfortable talking to their supervisors about mental health conditions
        """)
    st.divider()
    st.image("C:\\Users\\mitan\\Downloads\\subplots6.png", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        • A large number of respondents work in a tech company.
        """)
    with col2:
        st.markdown("""
        • It is observed that not a lot of them know about mental health benefits provided.
        """)
    st.divider()

    st.header("Bivariate Analysis")
    st.image("C:\\Users\\mitan\\Downloads\\subplots7.png",caption="Analyzing various columns wrt treatment column", use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        -- Gender vs Treatment: Males dominate the dataset but are less likely to seek treatment; females show higher treatment-seeking rates.\n
        -- No. of Employees vs Treatment: Both small (<100) and large (1000+) organizations show relatively higher treatment-seeking rates compared to mid-sized firms.
        """)
    with col2:
        st.markdown("""
        -- Age vs Treatment: Most respondents are between 25–35 years; treatment-seeking is relatively balanced across age groups.\n
        -- Country vs Treatment: Majority of responses come from the US, where treatment-seeking is higher than non-seeking.
        """)
    st.divider()
    st.image("C:\\Users\\mitan\\Downloads\\subplots8.png", use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        -- Family History vs Treatment: Those with a family history of mental illness are far more likely to seek treatment.
        """)
    with col2:
        st.markdown("""
        -- Mental vs Physical (by Age Group): Younger and middle-aged groups (20–39) mostly report uncertainty about whether mental health is taken as seriously as physical health.
        """)
    st.divider()
    st.image("C:\\Users\\mitan\\Downloads\\subplots9.png", use_container_width=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("""
        -- Is it easy to get a leave in a tech company?\n
           Most respondents, both in tech and non-tech companies, marked "Unknown" when asked about leave ease,
          but tech employees also had higher proportions reporting "Somewhat easy" and "Very easy" compared to non-tech employees.
        """)
    with col2:
        st.markdown("""
        -- Work interference vs. seeking treatment\n
           People who report frequent or occasional work interference (“Often” or “Rarely”) are more likely to seek treatment,
            whereas those reporting “Never” are predominantly not seeking treatment.
        """)
    st.divider()
    st.header("Multivariate Analysis")
    st.image("C:\\Users\\mitan\\Downloads\\subplots10.png",caption="Correlation Heatmap", use_container_width=True)
    st.markdown("""
        -- Most features show weak correlations, indicating low multicollinearity,
                 but notable positive correlations exist between supervisor and coworkers (0.57), wellness_program and seek_help (0.62), 
                and mental_health_consequence and phys_health_consequence (0.51). 
                Negative correlations include supervisor with mental_health_consequence (-0.50).
        """)
    footer()

def classification():
    st.title("🧮 Will the person seek treatment?")
    st.divider()
    st.markdown("We need to predict the employee would seek help or not, making it a binary classification task. Below are the models used, and their evaluation results")
    st.subheader("📌 Models Trained ")
    st.write(" - Logistic Regression\n - KNN Classification\n - XGB Classification\n - Random Forest Classification")
    st.markdown("""### **Logisitc Regression**""")
    st.write("A baseline linear classifier that predicts probabilities and works well for binary treatment prediction")
    st.code("Accuracy: 0.68\n AUC Score: 0.75")

    st.markdown("""### **KNN Classification**""")
    st.write("Classifies based on the majority label among the closest data points in feature space.")
    st.code("Accuracy: 0.67\n AUC Score: 0.71")
    st.image("C:\\Users\\mitan\\Downloads\\knn.png", caption="Accuracy vs K value", use_container_width=True)


    st.markdown("""### **XGB Classification**""")
    st.write("A powerful boosting-based classifier known for high accuracy, especially on structured and tabular data.")
    st.code("Accuracy: 0.74\n AUC Score: 0.812")


    st.markdown("""### **Random Forest Classification**""")
    st.write("An ensemble model that uses multiple decision trees to improve classification stability and accuracy.")
    st.code("Accuracy: 0.698\n AUC Score: 0.75\n F1 Score(weighted): 0.713")

    st.image("C:\\Users\\mitan\\Downloads\\auc.png", caption="Comparision Of Models", use_container_width=True)
    st.divider()
    st.markdown("### ✅ From the evaluation metrics it is evident that `XGB Classifiction` provides the best accuracy")

    st.divider()
    st.markdown("### 💬 A quick survey to predict whether a person is likely to seek mental health support or not ⬇️")
    input_dict_clf = {}
    display_names_clf = {
    "Gender": "Enter your Gender",
    "family_history": "Do you have a family history of mental illness?",
    "remote_work": "Do you work remotely (outside of an office) at least 50%% of the time?",
    "tech_company":"Do you work in a tech company?",
    "obs_consequence": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
    "work_interfere": "If you have a mental health condition, do you feel that it interferes with your work?",
    "coworkers": "Would you be willing to discuss a mental health issue with your coworkers?",
    "supervisor": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    "benefits": "Does your employer provide mental health benefits?",
    "seek_help": "Does your employer provide resources to learn more about mental health issues and how to seek help?",
    "anonymity": "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",
    "care_options": "Do you know the options for mental health care your employer provides?",
    "wellness_program": "Has your employer ever discussed mental health as part of an employee wellness program?",
    "mental_health_consequence": "Do you think that discussing a mental health issue with your employer would have negative consequences?",
    "phys_health_consequence": "Do you think that discussing a physical health issue with your employer would have negative consequences?",
    "mental_vs_physical":"Do you think mental health is taken as seriously as physical health?",
    "mental_health_interview":"Would you bring up a mental health issue with a potential employer in an interview? ",
    "phys_health_interview":"Would you bring up a physical health issue with a potential employer in an interview? "
    }
    cols=['family_history','Gender',
          'remote_work','tech_company','obs_consequence','work_interfere',
         'supervisor','coworkers','benefits','wellness_program','seek_help','anonymity',
          'mental_vs_physical','care_options','mental_health_consequence','phys_health_consequence',
         'mental_health_interview','phys_health_interview']
    # mapping for options in app

    ynmapping={1:'Yes',0:'No' }
    df[['family_history','remote_work','tech_company','obs_consequence']]=df[['family_history','remote_work','tech_company','obs_consequence']].replace(ynmapping)
    work_map = {'Often': 1, 'Sometimes': 0.75, 'Rarely': 0.25, 'Never': 0}
    reverse_work_map = {v: k for k, v in work_map.items()}
    df['work_interfere'] = df['work_interfere'].replace(reverse_work_map)
    col2map = {'Yes': 1, 'No': 0, 'Some of them': 0.5}
    reverse_col2map = {v: k for k, v in col2map.items()}
    df[['supervisor','coworkers']] = df[['supervisor','coworkers']].replace(reverse_col2map)
    othercols=['benefits','wellness_program','seek_help','anonymity','mental_vs_physical',
      'care_options','mental_health_consequence','phys_health_consequence',
      'mental_health_interview','phys_health_interview']
    cols_replace={1:'Yes', 0: 'No', 0.5:'Unknown/Maybe'}
    df[othercols]=df[othercols].replace(cols_replace)



    for col in df[cols]:
        options = df[col].dropna().unique().tolist()
        label = display_names_clf.get(col, col)
        input_dict_clf[col] = st.selectbox(label, options)

    input_df = pd.DataFrame([input_dict_clf])

    #reversing all mapping for predictions
    genmap={'Female':1, 'Male':0 , 'Others':0.5}
    input_df['Gender']=input_df['Gender'].replace(genmap)
    # Reverse mapping for Yes/No
    ynmapping_rev = {'Yes':1,'No':0}
    input_df[['family_history', 'remote_work', 'tech_company', 'obs_consequence']] = \
    input_df[['family_history', 'remote_work', 'tech_company', 'obs_consequence']].replace(ynmapping_rev)

    # Reverse mapping for work_interfere
    reverse_work_map = {'Often': 1, 'Sometimes': 0.75, 'Rarely': 0.25, 'Never': 0}
    input_df['work_interfere'] = input_df['work_interfere'].replace(reverse_work_map)

    # Reverse mapping for supervisor & coworkers
    reverse_col2map = {'Yes': 1, 'No': 0, 'Some of them': 0.5}
    input_df[['supervisor', 'coworkers']] = input_df[['supervisor', 'coworkers']].replace(reverse_col2map)

    # Reverse mapping for other columns
    othercols = [
    'benefits', 'wellness_program', 'seek_help', 'anonymity', 'mental_vs_physical',
    'care_options', 'mental_health_consequence', 'phys_health_consequence',
    'mental_health_interview', 'phys_health_interview'
    ]
    cols_replace_rev = {'Yes':1,'No':0, 'Unknown/Maybe':0.5}
    input_df[othercols] = input_df[othercols].replace(cols_replace_rev)


    if st.button("Predict Yes/No for treatment"):
        prediction = clf_model.predict(input_df)[0]

        if prediction == 1:
            st.success("✅ Model predicts that person will likely seek treatment!")
        else:
            st.error("❌ Model Predicts that person will likely not seek treatment!")

    footer()

def regression():
    st.title("🧮 What could be the approximate age of respondent")
    st.divider()
    st.markdown("The regression task aims to predict the `Age` of employees using various regression models. Below are the models used")

    st.subheader("📌 Models Trained")
    st.write(" - Linear Regression\n - Random Forest Regression\n - KNN Regression")

    st.markdown("""### **Linear Regression**""")
    st.write("Linear Regression is a simple and interpretable model that fits a linear relationship between the features and the target variable.")
    st.code("MAE: 5.27 \n" \
    "MSE: 46.94\n" \
    "R² Score: 0.0639")

    st.markdown("""### **Random Forest Regression**""")
    st.write("Random Forest Regressor is an ensemble method that builds multiple decision trees and averages their predictions for better accuracy and robustness.")
    st.code("MAE: 5.34 \n" \
    "MSE: 47.8 \n" \
    "R² Score: 0.046")

    st.markdown("""### **KNeighbours Regression**""")
    st.write("Predicts based on the average of predictions among the closest data points in feature space.")
    st.code("MAE:5.23 \n" \
    "MSE: 47.03\n" \
    "R² Score: 0.062")
    st.image("C:\\Users\\mitan\\Downloads\\knnr.png", caption="R2 score vs K value", use_container_width=False)

    st.markdown("### ✅ By looking at the evaluation metrics, we can see that `Linear Regression` performs the best and will be used for prediction purposes.")
    st.divider()
    st.markdown("### This is a sample predictor of the age of a person given their conditions ⬇️")

    input_dict_reg = {}
    display_names_reg =  {
    "Gender": "Enter your Gender",
    "family_history": "Do you have a family history of mental illness?",
    "remote_work": "Do you work remotely (outside of an office) at least 50%% of the time?",
    "tech_company":"Do you work in a tech company?",
    "obs_consequence": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
    "work_interfere": "If you have a mental health condition, do you feel that it interferes with your work?",
    "coworkers": "Would you be willing to discuss a mental health issue with your coworkers?",
    "supervisor": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    "benefits": "Does your employer provide mental health benefits?",
    "seek_help": "Does your employer provide resources to learn more about mental health issues and how to seek help?",
    "anonymity": "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",
    "care_options": "Do you know the options for mental health care your employer provides?",
    "wellness_program": "Has your employer ever discussed mental health as part of an employee wellness program?",
    "mental_health_consequence": "Do you think that discussing a mental health issue with your employer would have negative consequences?",
    "phys_health_consequence": "Do you think that discussing a physical health issue with your employer would have negative consequences?",
    "mental_vs_physical":"Do you think mental health is taken as seriously as physical health?",
    "mental_health_interview":"Would you bring up a mental health issue with a potential employer in an interview? ",
    "phys_health_interview":"Would you bring up a physical health issue with a potential employer in an interview? "
    }


    cols=['family_history','remote_work','Gender','tech_company','obs_consequence','work_interfere',
         'supervisor','coworkers','benefits','wellness_program','seek_help','anonymity',
          'mental_vs_physical','care_options','mental_health_consequence','phys_health_consequence',
         'mental_health_interview','phys_health_interview']
    # mapping for options in app

    ynmapping={1:'Yes',0:'No' }
    df[['family_history','remote_work','tech_company','obs_consequence']]=df[['family_history','remote_work','tech_company','obs_consequence']].replace(ynmapping)
    work_map = {'Often': 1, 'Sometimes': 0.75, 'Rarely': 0.25, 'Never': 0}
    reverse_work_map = {v: k for k, v in work_map.items()}
    df['work_interfere'] = df['work_interfere'].replace(reverse_work_map)
    col2map = {'Yes': 1, 'No': 0, 'Some of them': 0.5}
    reverse_col2map = {v: k for k, v in col2map.items()}
    df[['supervisor','coworkers']] = df[['supervisor','coworkers']].replace(reverse_col2map)
    othercols=['benefits','wellness_program','seek_help','anonymity','mental_vs_physical',
      'care_options','mental_health_consequence','phys_health_consequence',
      'mental_health_interview','phys_health_interview']
    cols_replace={1:'Yes', 0: 'No', 0.5:'Unknown/Maybe'}
    df[othercols]=df[othercols].replace(cols_replace)


    for col in cols:
        options = df[col].dropna().unique().tolist()
        label = display_names_reg.get(col, col) 
        input_dict_reg[col] = st.selectbox(label, options)

    input_df = pd.DataFrame([input_dict_reg])

    #reversing all mapping for predictions
    genmap={'Female':1, 'Male':0 , 'Others':0.5}
    input_df['Gender']=input_df['Gender'].replace(genmap)
    # Reverse mapping for Yes/No
    ynmapping_rev = {'Yes':1,'No':0}
    input_df[['family_history', 'remote_work', 'tech_company', 'obs_consequence']] = \
    input_df[['family_history', 'remote_work', 'tech_company', 'obs_consequence']].replace(ynmapping_rev)

    # Reverse mapping for work_interfere
    reverse_work_map = {'Often': 1, 'Sometimes': 0.75, 'Rarely': 0.25, 'Never': 0}
    input_df['work_interfere'] = input_df['work_interfere'].replace(reverse_work_map)

    # Reverse mapping for supervisor & coworkers
    reverse_col2map = {'Yes': 1, 'No': 0, 'Some of them': 0.5}
    input_df[['supervisor', 'coworkers']] = input_df[['supervisor', 'coworkers']].replace(reverse_col2map)

    # Reverse mapping for other columns
    othercols = [
    'benefits', 'wellness_program', 'seek_help', 'anonymity', 'mental_vs_physical',
    'care_options', 'mental_health_consequence', 'phys_health_consequence',
    'mental_health_interview', 'phys_health_interview'
    ]
    cols_replace_rev = {'Yes':1,'No':0, 'Unknown/Maybe':0.5}
    input_df[othercols] = input_df[othercols].replace(cols_replace_rev)


    # Predict Age
    if st.button("Predict the Age 🤞"):
        # transformed = reg_pre.transform(input_df)
        predicted_age = reg_model.predict(input_df)

        st.success(f"🎯 Here's what the model predicted : **{int(round(predicted_age[0]))} years**")


    footer()

def clustering():
    st.title("📊 Clustering and Persona Classification")
    st.divider()
    st.markdown("The objective of this task is to make clusters and group tech workers according to their mental health personas." \
    " Below are some of the techniques and algorithms applied for the same.")
   
    st.subheader("Techniques Used: ")
    st.write(" - Principal Component Analysis (PCA)\n - t-distributed Stochastic Neighbor Embedding (t-SNE)")
    st.write("Plots for the new componenets/features by these techniques-")
    col1,col2=st.columns(2)
    with col1:
        st.image("C:\\Users\\mitan\\Downloads\\pca.png", caption="PCA", use_container_width=False)
    with col2:
        st.image("C:\\Users\\mitan\\Downloads\\tsne.png", caption="t-SNE", use_container_width=False)

    st.markdown("From these clusters we can see that `t-SNE` forms the best clusters")
    st.subheader("Finding ideal number of clusters for different models: ")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("### Kmeans")
        st.image("C:\\Users\\mitan\\Downloads\\kmeanselb.png", caption="Elbow Method", use_container_width=False)
        st.write("Best number of clusters would be '6' ")
        st.image("C:\\Users\\mitan\\Downloads\\kmeans.png", caption="Clustering using Kmeans", use_container_width=False)
        st.code("Silhouette Score:  0.43073913")

    with col2:
        st.markdown("### Agglomerative")
        st.image("C:\\Users\\mitan\\Downloads\\agg.png", caption="Sihouette score comparision", use_container_width=False)
        st.write("Best number of clusters would be '3' ")
        st.image("C:\\Users\\mitan\\Downloads\\aggcluster.png", caption="Clustering using Agglomerative/Hierarchial", use_container_width=False)
        st.code("Silhouette Score: 0.45951703")

    st.markdown("### ✅ From these scores, we can easily see that 'Agglomerative Clustering' provides better results")
    st.subheader("Persona Classification ;)")
    st.markdown(""" ###  Cluster 0: Stable but Underinformed Allies""")
    st.write("Mostly in their mid-30s, these individuals show moderate mental stability at work and about half have received mental \
                health treatment. They often experience some work interference but still manage to function without severe disruption. \
                While they’re fairly comfortable talking to coworkers and supervisors about issues, they’re not deeply aware of workplace policies — \
                benefits, leave, and anonymity awareness are only moderate.Their limited policy knowledge means they’re not fully equipped to navigate \
                or advocate within formal systems.")

    st.markdown(""" ###  Cluster 1: Actively Struggling Policy-Aware Fighters""")
    st.write("Predominantly early 30s, ****this is the most treatment-engaged group**** — nearly all have received mental health care \
                and **many have a family history of mental health issues**. Work interference is high, yet they remain communicative with \
                supervisors and peers. They ****display the highest awareness of workplace benefits and anonymity protections****, likely due\
                to necessity and lived experience.Their blend of high need and high awareness makes them natural advocates,\
                but also at risk of burnout.")
    
    st.markdown(""" ###  Cluster 2:  Detached but Optimistic Independents""")
    st.write("In their early 30s on average, this group has ****no reported mental health treatment and minimal work interference****. \
                Few have a family history of mental health challenges, and they are moderately aware of workplace benefits and policies.\
                 They are somewhat open to conversations with supervisors but less connected with coworkers. While not actively disengaged,\
                 their passive stance means they are unlikely to seek or advocate for resources unless directly impacted.")
    
    footer()

pg = st.navigation([
  st.Page(Home, title="Dataset Visualization"),
  st.Page(EDA, title="Exploratory Data Analysis"),
  st.Page(classification, title="Classification Task"),
  st.Page(regression, title="Regression Task"),
  st.Page(clustering, title="Persona Clusters")
])
pg.run()

