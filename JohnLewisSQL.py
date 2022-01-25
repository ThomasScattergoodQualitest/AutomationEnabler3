import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
import matplotlib.pyplot as plt
from pandas import DataFrame
import mysql.connector
import numpy as np
import datetime
from sqlalchemy import *
from dateutil.relativedelta import relativedelta

#Creates a new table in sql and uploads data from csv into table 

#Change table name and csv file to uplaod new data



table_name = "johnlewistest"

#table_name = input('Enter the name of the table: ')
df = pd.read_csv(r'JLAllPages.csv')
#df.columns=['Review', 'Date', 'Subjectivity', 'Polarity', 'Score', 'Client']
df.columns=['Review', 'Date'] 




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
print(y_list)

days_list = [int(i) for i in y_list]
tday = datetime.datetime.today()
dtdelta = [tday - relativedelta(days=x) for x in days_list]

JLframe['Datetime'] = dtdelta
 

JLframe = JLframe.sort_values(by="Datetime")
JLframe['Year'] = JLframe['Datetime'].dt.year
JLframe['Package'] = np.random.randint(1, 5, JLframe.shape[0])
JLframe['Customer'] = np.random.randint(1, 5, JLframe.shape[0])
JLframe['Overall'] = np.random.randint(1, 5, JLframe.shape[0])
JLframe['Broadband'] = np.random.randint(1, 5, JLframe.shape[0])
#JLframe = JLframe.groupby('Year').agg({'Package': 'mean','Customer': 'mean','Overall': 'mean','Broadband': 'mean'}).reset_index()

# Plot of average rating per year
plt.plot(JLframe['Year'], JLframe['Package'], label ="package rating")
plt.plot(JLframe['Year'], JLframe['Customer'], label="customer support")
plt.plot(JLframe['Year'], JLframe['Broadband'], label="broadband speed")
plt.plot(JLframe['Year'], JLframe['Overall'], label="overall satisfaction ")
plt.title('Average Ratings per Year')
plt.xlabel('Date')
plt.ylabel('Rating score')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          ncol=3, fancybox=True, shadow=True)
plt.show()

# =============================================================================
# engine = create_engine('mysql://Thomas:Password2@localhost:3306/reviewdata', echo=True)
#  
# table_name = input('Enter the name of the table: ')
# JLframe.to_sql(table_name,engine)
# =============================================================================
