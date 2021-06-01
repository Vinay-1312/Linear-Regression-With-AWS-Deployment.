import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sklearn
data=pd.read_csv("Admission_Predict.csv")
print(data.head())

#checking is there any missing data
print(data.isnull().sum())

#there are no missing values
#dropping serial No as it wont be much coorelated to dependent feature

data=data.drop('Serial No.',axis=1)
print(data.head())

#plotting scatter plot for each independant feature with dependant
for i in data.columns:
    if i!='Chance of Admit ':
        plt.scatter(data[i],data['Chance of Admit '])
        plt.xlabel(i)
        plt.ylabel('Chance of Admit')
        plt.title(i+'VS chanece of admit')
        plt.show()

#We can see most of the relation are colinear

#plotting heatmap to get  correlation matrix
sns.set(rc={'figure.figsize':(11.7,6.29)})
correlation_matrix = data.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

#all the independant features are highly correlated with some.
#In some  independant features we can see multicolinartiy as well.
#This project is mainly focused on deplyment so we wont drop those features/.

from sklearn.model_selection import train_test_split
X=data.drop('Chance of Admit ',axis=1)
Y=data['Chance of Admit ']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
ypred=lr.predict(x_test)

from sklearn.metrics import r2_score
score=r2_score(ypred,y_test)
print(score)

pickle.dump(lr,open('model.pickle','wb'))