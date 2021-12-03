# Let's start with decision tree modeling to build an intuitive understanding of how we can split our data and focus our model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Import the csv and turn it into a dataframe
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

# Create a new column 'sex' where male = true and female = false
df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values


# Create the DecisionTreeClassifier object
model = DecisionTreeClassifier()

# Do a train/test split using a random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# Use the fit method to start training the model
model.fit(X_train, y_train)

status = model.predict([[3, True, 22, 1, 0, 7.25]])

if (status == 0):
    print('is dead')