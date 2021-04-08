# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 00:42:59 2021

@author: YonQi
"""

import streamlit as st
import streamlit.components.v1 as components



st.title('Customer Sentiment Analysis')  
#################Import Libraries###############
st.text(" \n")   
st.header('__Data preprocessing__')
st.text(" \n") 
st.write("- Import Libraries")
with st.echo():
    import pandas as pd
    import plotly.express as px
    import string
    import re



##################Read data##################
st.text(" \n")
st.write("- Import the dataset")
with st.echo():
    df = pd.read_csv("D:/Python Files/Datasets/Womens Clothing E-Commerce Reviews.csv")
    df.head()
   
       
def load_data():
    data = pd.read_csv("D:/Python Files/Datasets/Womens Clothing E-Commerce Reviews.csv")
    return data    

df = load_data()
st.write(df)


###############Number of rows,columns#########
st.text(" \n") 
st.write("- Get the number of rows and columns")
with st.echo():
    df.shape
    #rows X columns
    
    
    
###############Clean up the data###############   
st.text(" \n")
st.subheader("__Identify missing values__")
with st.echo():
    #Define new dataset - remove unwanted columns
    data = [df['Review Text'], df['Rating'],df['Age'],df['Department Name'] ] 
    
    df = pd.concat(data, axis=1)

    #Check for missing values of Review Text
    df.isnull().sum()
    
st.write(df.isnull().sum())

with st.echo():
    #Drop the particular rows which don't have any review
    df = df[~df['Review Text'].isnull()]

    #Get the new number of rows and columns        
    df.shape

    df['Review Text'] = df['Review Text'].astype(str)





################Exploratory data analysis###############
st.text(" \n") 
st.subheader('__Exploratory Data Analysis__')

with st.echo():
    import plotly.express as px
    
st.text(" \n") 
st.write('- Rating Distribution of Reviews ')
with st.echo():
    rating = df['Rating'].value_counts()
    rating = rating.reset_index()
    rating.columns = ['Rating','Count']
    fig = px.bar(rating, x="Rating", y="Count", color='Rating')
        
st.plotly_chart(fig)

st.text(" \n") 
st.write('- Age Distribution of Reviews ')
with st.echo():
    age = df['Age'].value_counts()
    age = age.reset_index()
    age.columns = ['Age','Count']
    fig_2 = px.bar(age, x="Age", y="Count", color='Age')
        
st.plotly_chart(fig_2)
    
st.text(" \n") 
st.write('- Department Name Distribution of Reviews ')    
with st.echo():
    dp_name = df['Department Name'].value_counts()
    dp_name = dp_name.reset_index()
    dp_name.columns = ['Department Name','Count']
    fig_4 = px.bar(dp_name, x="Department Name", y="Count", color='Department Name')

st.plotly_chart(fig_4)




############Clean up the text data###############
st.text(" \n")
st.text(" \n")
st.subheader("__Word Frequency Used__")

with st.echo():
    import nltk
    from nltk.corpus import stopwords
    from stop_words import get_stop_words
    from nltk import word_tokenize

st.text(" \n") 
st.write("- Remove stopwords")
st.write("- Remove punctuation")
st.write("- Remove characters length less than 2")
st.write("- Remove numbers")

with st.echo():
    a = df['Review Text'].str.lower().str.cat(sep=' ')
    
    #Removes punctuation,numbers and returns list of words
    b = re.sub('[^A-Za-z]+', ' ', a)

    #Remove all the stopwords from the text
    stop_words = list(get_stop_words('en'))         
    nltk_words = list(stopwords.words('english'))   
    stop_words.extend(nltk_words)

    #Split the string text into word tokenize
    word_tokens = word_tokenize(b)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    #Remove characters which have length less than 2  
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]

    #Remove numbers
    cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]        

st.text(" \n") 
st.write('- Top 20 Word Frequency Distribution') 
with st.echo():
    # Calculate and find the frequency of top 20 words
    word_dist = nltk.FreqDist(cleaned_data_title)
    rslt = pd.DataFrame(word_dist.most_common(20), columns=['Word', 'Frequency'])
    rslt = rslt.set_index('Word')
    
    fig_5 = px.bar(rslt)
        
st.plotly_chart(fig_5)






##################Vectorization########################
st.text(" \n") 
st.header('__Vectorization__')
with st.echo():
    def text_process(review):
        #Remove punctuation
        nopunc=[word for word in review if word not in string.punctuation]
        nopunc=''.join(nopunc)
        
        #Remove stopswords
        clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
        
        return clean_words
    
    

    
st.text(" \n") 
st.write('- Assumed the reviews which has 5 rating as positive and 1 rating as negative.')
with st.echo():
    rating_class = df[(df['Rating'] == 1) | (df['Rating'] == 5)]
    X_review = rating_class['Review Text']
    y = rating_class['Rating']
    


st.text(" \n")     
st.write('- Convert a collection of text to a vector of token counts')    
with st.echo():
    #Import CountVectorizer from sklearn library
    from sklearn.feature_extraction.text import CountVectorizer
    
    cv = CountVectorizer(analyzer=text_process)
    
    #Tokenize and build vocabulary
    cv.fit(X_review)
    
    #Encode document
    X_review = cv.transform(X_review)
    
    
    
#################Classification Model#####################
st.text(" \n") 
st.header('__Classification Model__')
st.subheader('__Split the dataset__')
with st.echo():
    #Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_review, y, test_size=0.20, random_state = 0)



st.text(" \n") 
alg = ['Naive Bayes Classifier', 'Random Forest Classifier']
classifier = st.selectbox('Select Algorithm', alg)


if classifier == 'Naive Bayes Classifier':
        
    st.subheader('__Train Algorithm__')
    with st.echo():
        #create and train the Naive Bayes Classifier
        from sklearn.naive_bayes import MultinomialNB
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
    
    st.text(" \n")     
    st.header('__Evaluation__')
    st.subheader('__Classification Report__')
    with st.echo():
        #Evaluate the model on the testing data set
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.metrics import plot_confusion_matrix
        
        nb_pred = nb.predict(X_test)
        
        st.write(classification_report(y_test, nb_pred))
    
    
    st.text(" \n")     
    st.subheader('__Confusion Matrix__')
    with st.echo():
        st.write(confusion_matrix(y_test, nb_pred))
        plot_confusion_matrix(nb, X_test, y_test)
        st.pyplot()
        
    st.set_option('deprecation.showPyplotGlobalUse', False)
        
    
    st.text(" \n")     
    st.subheader('__LIME Explainer__')
    with st.echo():
        from sklearn.pipeline import make_pipeline
        
        #Convert the vectoriser and model into a pipeline
        c = make_pipeline(cv, nb)
        
        #Create the LIME explainer
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=['negative','positive'])

        #Choose a random single prediction
        idx = 110
        exp = explainer.explain_instance(df['Review Text'][idx], c.predict_proba, num_features=6)

        exp.show_in_notebook(text=True)
        
    components.html(exp.as_html(), height = 800)
    



elif classifier == 'Random Forest Classifier':
    
    st.subheader('__Train Algorithm__')
    with st.echo():
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
    
    
    st.text(" \n") 
    st.header('__Evaluation__')
    st.subheader('__Classification Report__')
    with st.echo():
        #Evaluate the model on the testing data set
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.metrics import plot_confusion_matrix
        
        rf_pred = rf.predict(X_test)
        
        st.write(classification_report(y_test, rf_pred))
        
        
        
        
    st.text(" \n")     
    st.subheader('__Confusion Matrix__')
    with st.echo():
        st.write(confusion_matrix(y_test, rf_pred))
        plot_confusion_matrix(rf, X_test, y_test)
        st.pyplot()
        
        
    st.set_option('deprecation.showPyplotGlobalUse', False)    
        
    st.text(" \n")     
    st.subheader('__LIME Explainer__')
    with st.echo():
        from sklearn.pipeline import make_pipeline
        
        #Convert the vectoriser and model into a pipeline
        c = make_pipeline(cv, rf)
        
        #Create the LIME explainer
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=['negative','positive'])

        #Choose a random single prediction
        idx = 110
        exp = explainer.explain_instance(df['Review Text'][idx], c.predict_proba, num_features=6)

        exp.show_in_notebook(text=True)
        
    components.html(exp.as_html(), height = 800)
    
