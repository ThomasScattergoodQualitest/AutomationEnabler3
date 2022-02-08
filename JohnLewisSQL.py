import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
import matplotlib.pyplot as plt
from pandas import DataFrame
import mysql.connector
import numpy as np
import datetime
from sqlalchemy import *
from dateutil.relativedelta import relativedelta
from textblob import TextBlob
from nltk.tokenize import word_tokenize as wt, sent_tokenize as st
import pyLDAvis.gensim_models
import gensim.corpora
import gensim
from gensim.models import CoherenceModel
import nltk, re, string
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


#INIT DATA
stopList = list(nltk.corpus.stopwords.words('english'))
punct = string.punctuation +"``“”£"
contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

init_notebook_mode(connected=True)
cf.go_offline()


#CONTRACTION EXPANSION FUNCTION
def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x

#Creates a new table in sql and uploads data from csv into table 
#Change table name and csv file to uplaod new data

table_name = "johnlewistest"

#table_name = input('Enter the name of the table: ')
df = pd.read_csv(r'JLAllPages.csv')
#df.columns=['Review', 'Date', 'Subjectivity', 'Polarity', 'Score', 'Client']
df.columns=['Review', 'Date']

# =============================================================================
# df['Preview'] = list(map(lambda x: x[:254],df['Review']))
# 
# =============================================================================



#remove null values from dataframe
df.dropna(subset =["Date","Review"],inplace=True)



Base = declarative_base()



class Review(Base):
    __tablename__ = table_name+'Reviews'
    # Here we define columns for the table
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    review = Column(String(250), nullable=False)
    date = Column(DATETIME(), nullable=False)
    #preview = Column (String(250, unique=True))

    
    
# =============================================================================
# engine = create_engine('mysql://Thomas:Password2@localhost:3306/reviewdata', echo=True)
# 
# df.to_sql(table_name,engine)
# =============================================================================



mydb = mysql.connector.connect(user='Thomas', password='Password2',
                              host='localhost', database='reviewdata',
                              auth_plugin='mysql_native_password')
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM johnlewistest")
JLframe = mycursor.fetchall()
JLframe = DataFrame(JLframe,columns =['Review Id','Review', 'Date'])# 'Company'])


d = {'months': 31, 'years':365, 'year':365}
JLframe1 = JLframe['Date'].str.extract('(\d+)\s+(years|months|hours|days|year)', expand=True)
JLframe['Days'] = JLframe1[0].astype(float).mul(JLframe1[1].map(d)).astype('Int64').astype(str)
#JLframe['Unit'] = np.where(JLframe1[1].isin(['year', 'years','months', 'days']), ' days', ' ' + JLframe1[1])

y_list = JLframe['Days']
#print(y_list)

days_list = [int(i) for i in y_list]
tday = datetime.datetime.today()
dtdelta = [tday - relativedelta(days=x) for x in days_list]

JLframe['Datetime'] = dtdelta
 
# Insert dummy data for ratings - REMOVE IF DATA CONTAINS THESE RATINGS
JLframe = JLframe.sort_values(by="Datetime")
JLframe['Year'] = JLframe['Datetime'].dt.year
JLframe['Month'] = JLframe['Datetime'].dt.month
JLframe['Package'] = np.random.randint(1, 5, JLframe.shape[0])
JLframe['Customer'] = np.random.randint(1, 5, JLframe.shape[0])
JLframe['Overall'] = np.random.randint(1, 5, JLframe.shape[0])
JLframe['Broadband'] = np.random.randint(1, 5, JLframe.shape[0])
#JLframe = JLframe.groupby('Year').agg({'Package': 'mean','Customer': 'mean','Overall': 'mean','Broadband': 'mean'}).reset_index()

#group by average ratings per year
JLframe2 = JLframe.groupby('Year').agg({'Package': 'mean','Customer': 'mean','Overall': 'mean', 'Broadband': 'mean'}).reset_index()

#assign variables to the columns being plotted, DRY code
year = JLframe2['Year']
package=JLframe2['Package']
customer=JLframe2['Customer']
overall=JLframe2['Overall']
broadband=JLframe2['Broadband']
reviews = JLframe['Review']

# Polarity and Subjectivity 
comments=[]
polarities = []
subjectivities = []

for row in JLframe['Review']:
    comments.append(row)

for doc in comments: 
    blob = TextBlob(doc)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    polarities.append(polarity)
    subjectivities.append(subjectivity)
    #print(polarities)
    #print(subjectivities)

JLframe['Polarity'] = polarities
JLframe['Subjectivity'] = subjectivities

pol_mean = JLframe['Polarity'].mean()
sub_mean = JLframe['Subjectivity'].mean()
print ("average polarity before text cleaning:")
print(pol_mean)
print ("average subjectvitiy before text cleaning:")
print (sub_mean)


#CONTRACTION EXPANSION FOR BIGRAMS
#reviews_cont = JLframe['Review'].apply(lambda x: cont_to_exp(x))

#Tokenizing words
reviewWords = [wt(review) for review in reviews] # each review tokenized 
all_unfiltered = []
for doc in reviewWords:
    for word in doc:
        all_unfiltered.append(word) # all tokenized words
        
        
#Bigrams and Trigrams
bigram_phrases = gensim.models.Phrases(reviewWords, min_count=6, threshold=50)
trigram_phrases = gensim.models.Phrases(bigram_phrases[reviewWords],min_count=3, threshold=50)
bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram =  gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return(bigram[doc] for doc in texts)

def make_trigrams(texts):
    return(trigram[doc] for doc in texts)

#Making a bigrams list
data_bigrams = make_bigrams(reviewWords)
data_bigrams_trigrams = make_trigrams(data_bigrams)
bigrams_list = []
for i in data_bigrams:
    bigrams_list.append(i)
 


filtered_bigrams = [] # all reviews filtered into bigrams
for review in bigrams_list:
    filtered = [w.lower() for w in review if w not in stopList] # each filtered review
    filtered_bigrams.append(filtered)
    
    def clean_text_round1(text):
        #replace square brackets and content inside with ''
        text = re.sub('\[.*?\]', '', text)
        #remove instances of punctuation
        text = re.sub('[%s]' % re.escape(punct), '', text)
        #remove numbers and words attached to numbers
        text = re.sub('\w*\d\w*', '', text)

        return text
    
    clean_bigrams = []
    for doc in filtered_bigrams:
        doc = [clean_text_round1(word) for word in doc if word not in stopList]
        doc = list(filter(None, doc))
        #clean_bigram = pos_tag(doc)
        clean_bigrams.append(doc)
        
###############LEMMATIZER##############
def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)



lems = []
all_words = []

for doc in clean_bigrams:
    lem = [lemmatize_with_postag(word) for word in doc]
    for word in lem:
        if word not in stopList:
            all_words.append(word)
    lems.append(lem)

#JLframe['Lemmas'] = lems


#################PLOTS####################

# Plot of average rating per year
plt.plot(year, package, label ="package rating", color='b')
plt.plot(year, customer, label="customer support", color='r')
plt.plot(year, broadband, label="broadband speed", color='black')
plt.plot(year, overall, label="overall satisfaction", color='green')
plt.title('Average Ratings per Year')
plt.xlabel('Date')
plt.ylabel('Rating score')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          ncol=3, fancybox=True, shadow=True)
plt.show()

#bar plot of average ratings per year
plt.bar(year, package, label ="package rating", color='b', align='center', bottom=overall+broadband+customer)
plt.bar(year, customer, label="customer support", color='r', align='center', bottom=overall+broadband)
plt.bar(year, broadband, label="broadband speed", color='black', align='center',bottom=overall)
plt.bar(year, overall, label="overall satisfaction", color='green', align='center')
plt.title('Average Ratings per Year')
plt.xlabel('Date')
plt.ylabel('Rating score')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, fancybox=True, shadow=True)
plt.show()

#Scatter plot of subjectivness 
plt.scatter (polarities, subjectivities)
plt.xlabel ("polarities")
plt.ylabel ("subjectivities")
plt.title ("subjectivities as a function of polarities of John Lewis reviews")
plt.show



JLframe['Polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

# =============================================================================
# 
# JLList = JLframe['Preview'].values.tolist() 
# =============================================================================


# =============================================================================
# mycursor.execute("use reviewdata")
# mycursor.executemany("INSERT IGNORE INTO johnlewistest "
#                      "(Review, Date, Preview) "
#                      "VALUES (%s, %s, %s) ",JLList)
# mydb.commit()
# mydb.close()
# =============================================================================


# =============================================================================
# engine = create_engine('mysql://Thomas:Password2@localhost:3306/reviewdata', echo=True)
# table_name = "johnlewistest1"
# JLframe.to_sql(table_name,engine)
# 
# mydb.close()
# =============================================================================
