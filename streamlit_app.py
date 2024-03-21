import streamlit as st
import pandas as pd
import joblib
# Function to load the model
@st.cache_data
def load_model():
    with open('wine_quality_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

# Load your model
loaded_model = load_model()


# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Wine Quality Prediction Web App')


    # User inputs: textbox
    # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    # 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    # 'pH', 'sulphates', 'alcohol']
    fixed_acidity = st.text_input('Fixed Acidity', 7.4)
    volatile_acidity = st.text_input('Volatile Acidity', 0.7)
    citric_acid = st.text_input('Citric Acid', 0.0)
    residual_sugar = st.text_input('Residual Sugar', 1.9)
    chlorides = st.text_input('Chlorides', 0.076)
    free_sulfur_dioxide = st.text_input('Free Sulfur Dioxide', 11.0)
    total_sulfur_dioxide = st.text_input('Total Sulfur Dioxide', 34.0)
    density = st.text_input('Density', 0.9978)
    pH = st.text_input('pH', 3.51)
    sulphates = st.text_input('Sulphates', 0.56)
    alcohol = st.text_input('Alcohol', 9.4)
    
    
    user_inputs = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    
    if st.button('Predict'):
        df = pd.DataFrame(user_inputs, index=[0])
        # prediction = loaded_model.predict(df)
        prediction = [0]
        if prediction[0] == 1:
            st.success('Good Quality Wine')
        else:
            st.error('Bad Quality Wine')
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            # st.json(loaded_model.get_params())
            st.write('Model used: Random Forest Classifier')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'wine_quality_model.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="wine_quality_model.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'winequality-red.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="winequality-red.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Wine-Quality-Prediction)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is created to predict the quality of red wine based on the input features.')
    st.write('The model is trained using the Random Forest Classifier algorithm.')
    st.write('The dataset consists of 11 input features and 1 output feature. The input features are:')
    st.write('1. fixed acidity')
    st.write('2. volatile acidity')
    st.write('3. citric acid')
    st.write('4. residual sugar')
    st.write('5. chlorides')
    st.write('6. free sulfur dioxide')
    st.write('7. total sulfur dioxide')
    st.write('8. density')
    st.write('9. pH')
    st.write('10. sulphates')
    st.write('11. alcohol')
    st.write('The output feature is the quality of the wine, which is a binary variable (0 or 1).')
    
    
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Wine-Quality-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
