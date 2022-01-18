import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import mysql.connector


db = mysql.connector.connect(host="localhost", user="newuser", passwd="password123")
pointer = db.cursor()
pointer.execute("use reviewdata")
pointer.execute("SELECT * FROM johnlewisbroadband")
jlframe = pointer.fetchall()
jlframe = DataFrame(jlframe,columns =['Review Id','Review','Likes','Haha','Angry', 'Date', 'Company'])