import matplotlib.pyplot as plt
from pandas import DataFrame
import mysql.connector
import numpy as np
import datetime
from sqlalchemy import *
from dateutil.relativedelta import relativedelta


mydb = mysql.connector.connect(user='Thomas', password='Password2',
                              host='localhost', database='reviewdata',
                              auth_plugin='mysql_native_password')
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM johnlewisbroadband")
JLframe = mycursor.fetchall()
JLframe = DataFrame(JLframe,columns =['Review Id','Review', 'Date', 'Company'])


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


# =============================================================================
# x = JLframe.values.tolist()
# data = []
# for i in x:
#     i = tuple(i)
# data.append(i)
# 
# # =============================================================================
# # mycursor.execute("use reviewdata")
# # mycursor.execute("""ALTER TABLE johnlewisbroadband
# =============================================================================
# ADD COLUMN Days int AFTER Company,
# ADD COLUMN Datetime datetime,
# ADD COLUMN Year int,
# ADD COLUMN Package int,
# ADD COLUMN Customer int,
# ADD COLUMN Overall int,
# ADD COLUMN Broadband int """)
# 
# 
# add_data = ("INSERT INTO johnlewisbroadband "
# "(Review_Id, Review, Date, Datetime Company, Days, Year, Package, Customer, Overall, Broadband ) "
# "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, )")
# 
# mycursor.executemany(add_data,data)
# mydb.commit()
# 
# mycursor.close()
# mydb.close()
# 
# =============================================================================
