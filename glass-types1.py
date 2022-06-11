# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix
 
# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()
st.sidebar.subheader("Scatter Plot")

# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect("Select the x-axis values:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)

for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()
# Create histograms for all the features.
# Sidebar for histograms.
st.sidebar.subheader("Histogram")

# Choosing features for histograms.
hist_features = st.sidebar.multiselect("Select features to create histograms:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create histograms.
for feature in hist_features:
    st.subheader(f"Histogram for {feature}")
    plt.figure(figsize = (12, 6))
    plt.hist(glass_df[feature], bins = 'sturges', edgecolor = 'black')
    st.pyplot() 

# Create box plots for all the columns.
# Sidebar for box plots.
st.sidebar.subheader("Box Plot")

# Choosing columns for box plots.
box_plot_cols = st.sidebar.multiselect("Select the columns to create box plots:",
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))

# Create box plots.
for col in box_plot_cols:
    st.subheader(f"Box plot for {col}")
    plt.figure(figsize = (12, 2))
    sns.boxplot(glass_df[col])
    st.pyplot()

st.sidebar.subheader('Scatter Plot')
st.set_option('deprecation.showPyplotGlobalUse', False)
ftr = st.sidebar.multiselect('Select the xaxis values:', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for i in ftr:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=i, y = 'GlassType', data=glass_df)
    st.pyplot()
st.sidebar.subheader('Visualization Selector')
plot_types = st.sidebar.multiselect('Select the charts/plot:',('Histogram', 'Boxplot', 'Countplot', 'Piechart', 'Correlation_heatmap', 'Pairplot'))
if 'Histogram' in plot_types:
    st.sidebar.subheader('Histogram:')
    val = st.sidebar.multiselect('Select the values for Histogram:', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    plt.figure(figsize=(10, 5))
    plt.hist(glass_df[[val]], bins='sturges', edgecolor='orange')
    st.pyplot()
if 'Boxplot' in plot_types:
    st.sidebar.subheader('Boxplot:')
    val2 = st.sidebar.multiselect('Select the values for Boxplot:', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    plt.figure(figsize=(10, 5))
    sns.boxplot(glass_df[[val2]], bins='sturges', edgecolor='orange')
    st.pyplot()
if 'Countplot' in plot_types:
    st.sidebar.subheader('Countplot:')
    plt.figure(figsize=(10, 5))
    sns.countplot(glass_df['GlassType'])
    st.pyplot()
if 'Piechart' in plot_types:
    st.sidebar.subheader('Boxplot:')
    plt.figure(dpi = 100)
    plt.pie(glass_df['GlassType'].value_counts(), labels=glass_df['GlassType'].value_counts().index, autopct = '%1.1f%%')
    st.pyplot()
if 'Correlation_heatmap' in plot_types:
    st.sidebar.subheader('Correlation_heatmap:')
    plt.figure(dpi = 100)
    sns.heatmap(glass_df.corr(), annot=True, vmin=-1, vmax=1)
    st.pyplot()
if 'Pairplot' in plot_types:
    st.sidebar.subheader('Pairplot:')
    plt.figure(dpi = 100)
    sns.pairplot(glass_df)
    st.pyplot()
st.sidebar.subheader('Select your values:')
ri = st.sidebar.slider('Input RI:', float(glass_df['RI'].min()),float(glass_df['RI'].max()))
Mg = st.sidebar.slider('Input Mg:', float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
Al = st.sidebar.slider('Input Al:', float(glass_df['Al'].min()),float(glass_df['Al'].max()))
Na = st.sidebar.slider('Input Na:', float(glass_df['Na'].min()),float(glass_df['Na'].max()))
Si = st.sidebar.slider('Input Si:', float(glass_df['Si'].min()),float(glass_df['Si'].max()))
Ca = st.sidebar.slider('Input Ca:', float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
Ba = st.sidebar.slider('Input Ba:', float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
Fe = st.sidebar.slider('Input Fe:', float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))
K = st.sidebar.slider('Input K:', float(glass_df['K'].min()),float(glass_df['K'].max()))

st.sidebar.subheader('Choose Classifier:')
clsf = st.sidebar.selectbox('Classifier:', ('Support Vector Machine', 'Random Forest Classifier', "LogisticRegression"))
if clsf=='Support Vector Machine':
    st.sidebar.subheader('Model Hyperparameters')
    c = st.sidebar.number_input('c (error rate)', 1, 100, step=1)
    k = st.sidebar.radio('kernel', ('linear', 'rbf', 'poly'))
    gm = st.sidebar.number_input('Gamma....', 0,100, step = 1)
    if st.sidebar.button('Classify'):
        m = SVC(kernel = k, gamma = gm, C=c)
        m.fit(X_train, y_train)
        tsprd = m.predict(X_test)
        mscr = m.score(X_train, y_train)
        glstyp = prediction(m, ri, na, mg, al, si, ca, ba, fe, k)
        st.write('The glass type predicted is:', glstyp)
        st.write('Accuracy:', mscr)
        st.pyplot()
if clsf=='RandomForestClassifier':
    st.sidebar.subheader('Model Hyperparameters')
    nt = st.sidebar.number_input('Number of trees in the forest', 1, 100, step=1)
    mxdpt = st.sidebar.number_input('Maximum depth of one tree', 0,100, step = 1)
    if st.sidebar.button('Classify'):
        m2 = RandomForestClassifier(n_estimators = nt, max_depth = mxdpt)
        m2.fit(X_train, y_train)
        tsprd2 = m2.predict(X_test)
        mscr2 = m2.score(X_train, y_train)
        glstyp2 = prediction(m2, ri, na, mg, al, si, ca, ba, fe, k)
        st.write('The glass type predicted is:', glstyp2)
        st.write('Accuracy:', mscr2)
        st.pyplot()
if clsf=='LogisticRegression':
    st.sidebar.subheader('Model Hyperparameters')
    c = st.sidebar.number_input('C', 1, 100, step=1)
    mxit = st.sidebar.number_input('Maximum iteration', 0,100, step = 1)
    if st.sidebar.button('Classify'):
        m3 = LogisticRegression(C=c, max_iter = mxit)
        m3.fit(X_train, y_train)
        tsprd3 = m3.predict(X_test)
        mscr3 = m3.score(X_train, y_train)
        glstyp3 = prediction(m3, ri, Na, Mg, Al, Si, Ca, Ba, Fe, K)
        st.write('The glass type predicted is:', glstyp3)
        st.write('Accuracy:', f'{mscr3*100}%')
        st.pyplot()