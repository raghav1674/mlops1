from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
#new
model=LinearRegression() #creating the model

dataset=pd.read_csv("/my_dlws/all_storage/Salary_Data.csv") #loading csv file
x=dataset["YearsExperience"]
y=dataset["Salary"]

X=x.values.reshape(-1,1) #convert to 2d numpy array

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=20)



model.fit(X_train,y_train)  #training the model


X_test=sc.transform(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test.values,y_pred))

