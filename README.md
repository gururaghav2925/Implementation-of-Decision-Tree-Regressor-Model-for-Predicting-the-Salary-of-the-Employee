# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import Required Libraries**  
   Import `pandas`, `sklearn`, and other necessary libraries like `DecisionTreeRegressor`.

2. **Load and Preprocess the Dataset**  
   - Load the dataset using `pandas.read_csv()`  
   - Check for missing values and data types  
   - Encode categorical columns (e.g., `Position`) using `LabelEncoder`

3. **Define Features and Target Variable**  
   - Select relevant features such as `Position` and `Level`  
   - Define the target variable as `Salary`

4. **Split the Dataset**  
   - Use `train_test_split` to divide the data into training and testing sets

5. **Train and Evaluate the Model**  
   - Initialize and train a `DecisionTreeRegressor` on the training set  
   - Predict on the test set and evaluate using R² score (`r2_score`)  
   - Use `.predict()` to make salary predictions for specific inputs
 

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Guru Raghav Ponjeevith V
RegisterNumber:  212223220027
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
**data.head()**

![image](https://github.com/user-attachments/assets/9f8aa9c7-070e-486b-88d3-8030401b6ae7)



 **data.info()**

 ![image](https://github.com/user-attachments/assets/504d48c0-118c-4117-9824-f288316a22ee)


**LabelEncoder_fit**

![image](https://github.com/user-attachments/assets/adc21454-10dd-4a6e-abb6-c2a208c7b413)

**Position&Level**

![image](https://github.com/user-attachments/assets/004f078d-fa58-441f-b93d-c3b64a289a85)

**Salary**

![image](https://github.com/user-attachments/assets/b8e23d1e-e490-457d-b8b7-6b569e072011)


**Prediction**

![image](https://github.com/user-attachments/assets/dc2ebcec-1c1a-4711-a231-ff3f9eb4a710)

**Metric_Score**

![image](https://github.com/user-attachments/assets/37e7cdd0-2df0-4294-aaa9-74163ff109c6)


**Given Prediction**

![image](https://github.com/user-attachments/assets/0e6d87e1-02dd-4188-838b-052fb1b17bba)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
