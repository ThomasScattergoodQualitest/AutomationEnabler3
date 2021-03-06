import pandas as pd
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import matplotlib.pyplot as plt
from pandas import DataFrame
import mysql.connector
import numpy as np
from textblob import TextBlob


table_name = "kfctest"
#table_name = input('Enter the name of the table: ')
#df = pd.read_csv(r'KFCFinal.csv')
#df.columns=['Review', 'Like', 'Haha', 'Angry', 'Date', 'Subjectivity', 'Polarity','Score', 'Client']


#Reading in csv from selenium, this data will be exported to SQL
df = pd.read_csv(r'KFCReviews.csv')
#Define dataframe columns
df.columns=['Review', 'Like', 'Haha', 'Angry', 'Date']



#remove null values from dataframe
df.dropna(subset =["Date","Review"],inplace=True)



Base = declarative_base()



class Review(Base):
    __tablename__ = table_name+'Reviews'
    # Here we define columns for the table
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    #site_id = Column(Integer, nullable=False)
    review = Column(String(250), nullable=False)
    #rating = Column(String(250))
    like = Column (Integer, nullable = True)
    haha = Column (Integer, nullable = True)
    angry = Column (Integer, nullable = True)
    date = Column(DATETIME(), nullable=False)
    polarity = Column(FLOAT)
    subjectivity = Column(FLOAT)
    #dominant_topic = Column(Integer)
    #topic_contribution = Column(FLOAT)

    
#Below 2 lines create table in SQL using table name variable above
# =============================================================================
# engine = create_engine('mysql://Thomas:Password2@localhost:3306/reviewdata', echo=True)
# df.to_sql(table_name,engine)
# =============================================================================


#Connecting to created table 
db = mysql.connector.connect(host="localhost", user="Thomas", passwd="Password2", auth_plugin='mysql_native_password')
pointer = db.cursor()
pointer.execute("use reviewdata")
pointer.execute("SELECT * FROM facebook")

# Creting dataframe of data from SQL table 
kfcframe = pointer.fetchall()
kfcframe = DataFrame(kfcframe,columns =['Review Id','Review','Likes','Haha','Angry', 'Date', 'Company'])
kfcframe = kfcframe.drop(kfcframe.index[[0, 1, 3, 10,  11, 12, 21, 25, 28, 30, 32]])
kfcframe["Likes"] = pd.to_numeric(kfcframe["Likes"])
kfcframe["Haha"] = pd.to_numeric(kfcframe["Haha"])
kfcframe["Angry"] = pd.to_numeric(kfcframe["Angry"])
kfcframe['Date'] = pd.to_datetime(kfcframe['Date'])
kfcframe = kfcframe.sort_values(by="Date")

# the date must be changed into a date format so that it will be easier for plotting
#kfcframe['DATE'] = pd.to_datetime(kfcframe['Date'])
kfcframe['Day'] = kfcframe['Date'].dt.day
kfcframe['Month'] = kfcframe['Date'].dt.month
kfcframe['Year'] = kfcframe['Date'].dt.year
kfcframe['Length'] = kfcframe['Review'].apply(len)


#Making a list containing each review comment 
comments=[]
for row in kfcframe['Review']:
    comments.append(row)
#JLframe['Review']
#del comments[0]
#print(comments[0])


#Making lists of polarities and subjectivities
polarities = []
subjectivities = []

for doc in comments: 
    blob = TextBlob(doc)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    polarities.append(polarity)
    subjectivities.append(subjectivity)
    #print(polarities)
    #print(subjectivities)


#creating columnns in the dataframe for newly generated values  
kfcframe['Polarity'] = polarities
kfcframe['Subjectivity'] = subjectivities

pol_mean = kfcframe['Polarity'].mean()
sub_mean = kfcframe['Subjectivity'].mean()
print ("average polarity of KFC reviews:")
print(pol_mean)
print ("average subjectvitiy of KFC reviews:")
print (sub_mean)

####################PLOTS###################

#plot of review likes as a funcion of review length
plt.scatter(kfcframe['Length'], kfcframe['Likes'])
plt.title("Review Likes as a function of review length")
plt.xlabel("Review length (words)")
plt.ylabel("Amount of likes")
plt.show()

# =============================================================================
# #plot of likes per day in december  
# plt.plot(kfcframe['Day'], kfcframe['Likes'])
# plt.title("Number of likes in december as a function of time")
# plt.xlabel("Date")
# plt.ylabel("Number of likes")
# plt.show()
# =============================================================================


kfcframe2 = kfcframe.groupby('Day').agg({'Likes': 'sum','Haha': 'sum','Angry': 'sum'}).reset_index()
# insert dummy data for haha and angry reacts - REMOVE IF DATA CONTAINS THESE RATINGS
kfcframe2['Haha'] = np.random.randint(1, 5, kfcframe2.shape[0])
kfcframe2['Angry'] = np.random.randint(1, 5, kfcframe2.shape[0])


#assigning variables to dataframe columns, as to avoid dry code
likes=kfcframe2['Likes']
haha=kfcframe2['Haha']
angry=kfcframe2['Angry']
day=kfcframe2['Day']


#plot of average reactions over a month (line graph form)
plt.plot(day, likes, color='b', label="Likes")
plt.plot(day, haha, color='g', label="Haha")
plt.plot(day, angry, color='r', label="Angry")
plt.title('Average Reactions over a month (december)')
plt.xlabel('DATE')
plt.ylabel('Reaction count')
plt.legend()
plt.show()



#stacked bar plot of average reactions over a month
plt.bar(day, likes, color='b', align='center', label="Likes", bottom=angry+haha)
plt.bar(day, haha, color='g', align='center', label="Haha", bottom=angry)
plt.bar(day, angry, color='r', align='center', label="Angry")
plt.title("Number of total reactions per day in December")
plt.xlabel("Day in december")
plt.ylabel("Number of reactions")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, fancybox=True, shadow=True)
plt.show()


plt.scatter (polarities, subjectivities)
plt.xlabel ("polarities")
plt.ylabel ("subjectivities")
plt.title ("subjectivities as a function of polarities of KFC reviews")
plt.show

 
# =============================================================================
# engine = create_engine('mysql://Thomas:Password2@localhost:3306/reviewdata', echo=True)
# table_name2 = "kfctest1"
# # input('Enter the name of the table: ')
# kfcframe.to_sql(table_name2,engine)
# 
# =============================================================================


db.close()