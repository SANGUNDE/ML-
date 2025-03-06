import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pandas.read_csv("C:/Users/hp/Downloads/canada_per_capita_income.csv")

reg = LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'] )

m = reg.coef_
c = reg.intercept_

#very simple equation for calculating the predictions.
#y =mx + c
#replace the required details and kick-off your career in machine learning
y = m * 2020 + c
print(y)

#if you want to view the new model with the best fit line.
plt.scatter(df['year'], df['per capita income (US$)'])


#Repalce the reg.year and the actual year. I will upload a csv also 

plt.show()



