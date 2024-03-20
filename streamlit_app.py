import streamlit as st
import pandas as pd
import joblib
# Function to load the model
@st.cache_data
def load_model():
    with open('Titanic_survival_prediction_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

# Load your model
loaded_model = load_model()


# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Titanic Survival Prediction Web App')


    # User inputs: textbox
    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    Pclass = st.selectbox('Pclass', [1, 2, 3])
    sex = st.selectbox('Sex',["Male",'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    sibsp = st.number_input('SibSp', min_value=0, max_value=10, value=0)
    parch = st.number_input('Parch', min_value=0, max_value=10, value=0)
    fare = st.number_input('Fare', min_value=0, max_value=1000, value=50)
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
    
    # df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
    user_inputs = {
        'Pclass': Pclass,
        'Sex': 0 if sex == 'Male' else 1,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': 0 if embarked == 'S' else 1 if embarked == 'C' else 2
    }
    
    if st.button('Predict'):
        df = pd.DataFrame(user_inputs, index=[0])
        prediction = loaded_model.predict(df)
        if prediction[0] == 1:
            st.success('Survived')
        else:
            st.error('Not Survived')
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(loaded_model.get_params())
            st.write('Model used: Logistic Regression')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'titanic_model.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="titanic_model.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'train.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="titanic_survival_data.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Titanic-Survival-Prediction)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is a simple Titanic Survival Prediction web app. The web app uses a logistic regression model to predict whether the mail is spam or not.')
    
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Titanic-Survival-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
