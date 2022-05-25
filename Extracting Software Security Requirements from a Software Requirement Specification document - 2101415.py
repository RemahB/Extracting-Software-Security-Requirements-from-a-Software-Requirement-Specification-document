#!/usr/bin/env python
# coding: utf-8

# # **Extracting Software Security Requirements from a Software Requirement Specification document**

# In[1]:


pip install tabulate


# In[2]:


# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# load the data
df = pd.read_csv("SEDataset.csv", error_bad_lines=False, header=None, names=['text', 'label'])


# In[4]:


# lets try to look the data
df.head()


# In[5]:


# lets try to check the distribution of label
print(df['label'].value_counts())


# * We can see above , we have irrelevant label so lets remove these labels.

# In[6]:


df = df[df['label'].isin(['Confidentiality','Integrity','Availability'])]


# In[7]:


# Also visualize our target variable in PIE chart form
# Creating explode data
explode = (0.1, 0.0, 0.1)
# Creating color parameters
colors = ( "green", "red", "blue")
  
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "green" }


df['label'].value_counts().plot(kind="pie", figsize=(5,5),startangle=90,shadow=True,autopct="%1.1f%%",explode = explode,
                                   colors=colors,wedgeprops = wp)
plt.show()


# * From above anaylsis, only 4.2% is Availability that is imbalance form.

# In[8]:


df['length_text'] = df['text'].str.len()
sns.distplot(df['length_text'], color="r")
plt.show()


# In[9]:


get_ipython().system('pip install WordCloud')


# In[10]:


# importing the NLP libraries that will be used for preprocessing
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer() #creating an instance of the class

stopwords.words("english")[:10] # <-- import the english stopwords


# In[11]:


def preprocess_text(text):
    """In preprocess_text function we will apply all the things that given below:
    - removing links
    - removing special characters
    - removing punctuations
    - removing numbers
    - removing stopwords
    - doing stemming
    - transforming in lowercase
    - removing excessive whitespaces
    """
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove punctuations in string
    text = re.sub(r'[^\w\s]', "", text) 
    
    # remove stopwords, doing stemming
    # 1. tokenize
    tokens = nltk.word_tokenize(text)
    # 2. check if stopword and lemmatizing the word
    tokens = [lemma.lemmatize(w) for w in tokens if not w.lower() in stopwords.words("english") if len(w)>=3]
    # 3. join back together
    text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text


# In[12]:


df['cleaned'] = df['text'].apply(lambda x: preprocess_text(x))


# In[13]:


def numeric_label(x):
    if x == 'Availability':return 0
    elif x == 'Integrity':return 1
    else:return 2


# In[14]:


# lets convert categorical label into numeric format
df['Label'] = df['label'].apply(lambda x: numeric_label(x))


# In[15]:


# lets Preview the cleaned dataset again
df.head()


# In[16]:


# define a function for getting all words from the text
def returning_tokinize_list(df,column_name):
    df = df.reset_index(drop=True)  
    tokenize_list = [word_tokenize(df[column_name][i]) for i in range(df.shape[0])]
    final = [j for i in tokenize_list for j in i]
    return final     


# In[17]:


# get the all words of text into list
tokenize_list_words= returning_tokinize_list(df, 'cleaned')


# In[18]:


# function for words in dataframe format
def table_format(data_list,column_name):
    df_ = pd.DataFrame(data_list, columns = [column_name,'Frequency_distribution'])
    return df_

# function for extracting the most common words in reviews text
def most_common_words(cleaned_col_name_list,common_words = 10):
    fdist = FreqDist(cleaned_col_name_list)
    most_common=fdist.most_common(common_words)
    return most_common


# In[19]:


# plotting the graph of most common words
def frequency_dis_graph(cleaned_col_name_list,num_of_words=10):
    fdist = FreqDist(cleaned_col_name_list)
    fdist.plot(num_of_words,cumulative=False, marker='o')
    plt.show()

# draw a graph of word which are most common
def word_cloud(data):
    unique_string=(" ").join(data)
    wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("wordCloud"+".png", bbox_inches='tight')
    plt.show()
    plt.close()


# In[20]:


# draw word cloud
word_cloud(tokenize_list_words)


# In[21]:


# lets try to check the 15 most common words
MCW = most_common_words(tokenize_list_words, common_words=15)
table_format(MCW, 'word')


# In[22]:


# graph for showing top 15 most common words
frequency_dis_graph(tokenize_list_words,num_of_words=15)


# In[23]:


# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[24]:


# lets get dependent and independent features
X = df['cleaned']
y = df['Label']


# In[25]:


#split data into 70% training and 30% testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[26]:


# Fit and transform the training data to a document-term matrix using TfidfVectorizer 
tfidf = TfidfVectorizer()

x_train_tfidf = tfidf.fit_transform(x_train.values)
print ("Number of features : %d" %len(tfidf.get_feature_names()))


# In[27]:


# function for evaluation metrics precision, recall, f1 etc
def modelEvaluation(predictions, y_test_set, model_name):
    # Print model evaluation to predicted result    
    print("="*100)
    print("\t\t\t{}".format(model_name))
    print("="*100)
    
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test_set, predictions)))
    print ("Precision on validation set: {:.4f}".format(precision_score(y_test_set, predictions, average='macro')))    
    print ("Recall on validation set: {:.4f}".format(recall_score(y_test_set, predictions, average='macro')))
    print ("F1_Score on validation set: {:.4f}".format(f1_score(y_test_set, predictions, average='macro')))
    print ("\nClassification report : \n", classification_report(y_test_set, predictions, target_names=['Availability','Integrity','Confidentiality']))
    print ("\nConfusion Matrix : \n", confusion_matrix(y_test_set, predictions))
    sns.set(font_scale=1)
    cm = confusion_matrix(y_test_set, predictions)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(3, 3)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    results = [accuracy_score(y_test_set, predictions),precision_score(y_test_set, predictions, average='macro'),
              recall_score(y_test_set, predictions, average='macro'),f1_score(y_test_set, predictions, average='macro')]
    return results


# ## Model Building - Logistic Regression

# In[28]:


lr = LogisticRegression()
lr.fit(x_train_tfidf, y_train)
predictions = lr.predict(tfidf.transform(x_test))
results_lr = modelEvaluation(predictions, y_test, "logistic Regression")


# ## Model Building - SVM

# In[29]:


svc = SVC()
svc.fit(x_train_tfidf, y_train)
predictions = svc.predict(tfidf.transform(x_test))
results_svc = modelEvaluation(predictions, y_test, "SVM")


# ## Model Building - Multinomial Navie Bayes

# In[30]:


gnb = MultinomialNB()
gnb.fit(x_train_tfidf, y_train)
predictions = gnb.predict(tfidf.transform(x_test))
results_gnb = modelEvaluation(predictions, y_test, "Multinomial NB")


# ## Model Building - Random Forest

# In[31]:


rf = RandomForestClassifier()
rf.fit(x_train_tfidf, y_train)
predictions = rf.predict(tfidf.transform(x_test))
results_rf = modelEvaluation(predictions, y_test, "RandomForestClassifier")


# ## Model performance evaluation

# In[32]:


# showing all models result
dic = {
    'Metrics':['accuracy','precision','recall','f1-score'],
    'Logistic Regression' : results_lr,
    'SVM' : results_svc,
    'MultinomialNB' : results_gnb,
    'Random Forest' : results_rf,

}
metrics_df = pd.DataFrame(dic)

metrics_df = metrics_df.set_index('Metrics')
# displaying the DataFrame
print(tabulate(metrics_df.T, headers = 'keys', tablefmt = 'psql'))


# In[33]:


metrics_df.T.plot(kind='bar', figsize=(15,8))
plt.show()


# In[34]:


models_acc = metrics_df.iloc[0].tolist()
models_names = metrics_df.columns.tolist()
plt.figure(figsize=(15,5))
plt.bar(models_names,models_acc)
plt.title("Models Accuracy", fontsize=18)

def addLabels(names,acc):
    for i in range(len(names)):
        plt.text(i,round(acc[i],2),f"{round(acc[i],2)}%", ha = 'center', bbox=dict(facecolor='yellow', alpha=0.9))

addLabels(models_names,models_acc)
plt.show()


# In[ ]:





# In[ ]:




