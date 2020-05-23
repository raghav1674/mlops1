from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
#new
model=LinearRegression() #creating the model

dataset=pd.read_csv("/my_mlws/Salary_Data.csv") #loading csv file
x=dataset["YearsExperience"]
y=dataset["Salary"]

X=x.values.reshape(-1,1) #convert to 2d numpy array

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

model.fit(X_train,y_train)  #training the model
y_pred=model.predict(X_test)

print(mean_squared_error(y_test,y_pred))





# # VISUALIZING
import matplotlib.pyplot as plt
plt.scatter(X_test,y_test,label="actual_datapoint")
plt.plot(X_test,y_pred,label="predicted_datapoints")
plt.grid(True,color="g",linestyle="--")
# plt.xlim()
# plt.ylim(39343.00,67938.00)
plt.title("Testing_model")
plt.xlabel("Year of Joining")
plt.ylabel("Salary")


plt.legend()
plt.show()
plt.savefig("visuals/salary-estimator1.jpg",dpi=400)

# storing the model in storage

import  joblib
joblib.dump(model, "../created_models/my_salary-estimator.pkl1")