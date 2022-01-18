import pandas as pd
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship



table_name = input('Enter the name of the table: ')
df = pd.read_csv(r'NetflixFinal.csv')
df.columns=['Date','Review', 'Rating', 'Subjectivity', 'Polarity','Score']




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
    
engine = create_engine('mysql://Thomas:Password2@localhost:3306/reviewdata', echo=True)


df.to_sql(table_name,engine)