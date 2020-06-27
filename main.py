import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('train.csv')
df = df.drop(["Name", "Ticket", "SibSp", "Parch"], axis=1)
df = df.drop(['Cabin'], axis = 1)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
# print(df)

df1 = df[df['Survived'] == 0]
df2 = df[df['Survived'] == 1]

df1["Age"] = df1["Age"].fillna(value=df1["Age"].mean())
df2["Age"] = df2["Age"].fillna(value=df2["Age"].mean())

frames = [df1, df2]
df = pd.concat(frames)

df = df.sort_values(by='PassengerId')
print(df)

scaler = MinMaxScaler(feature_range=(0, 5))
df['Age'] = scaler.fit_transform(df[['Age']])
df['Fare'] = scaler.fit_transform(df[['Fare']])
# print(df)

X = df.iloc[:, 2:]
Y = df.iloc[:, 1]
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print(X_train)

k_range = range(1, 70)
scores = {}
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(Y_test, y_pred)
    scores_list.append(scores[k])

for k in k_range:
    print(f"{k} score:{scores[k]}")

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(Y_test, y_pred))
print(f"Best KNN found: {scores[4]}")
# k = 4  for KNN is found to be the best model accuracy-wise
df = pd.read_csv('test.csv')
df = df.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
df = df.drop(['Cabin'], axis = 1)

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

df1 = df[df['Pclass'] == 1]
df2 = df[df['Pclass'] == 2]
df3 = df[df['Pclass'] == 3]

df1["Age"] = df1["Age"].fillna(value=df1["Age"].mean())
df2["Age"] = df2["Age"].fillna(value=df2["Age"].mean())
df3["Age"] = df3["Age"].fillna(value=df3["Age"].mean())
frames = [df1, df2, df3]
df = pd.concat(frames)
df = df.sort_values(by='PassengerId')
df["Fare"] = df["Fare"].fillna(value=df["Fare"].mean())

scaler = MinMaxScaler(feature_range=(0, 5))
df['Age'] = scaler.fit_transform(df[['Age']])
df['Fare'] = scaler.fit_transform(df[['Fare']])

final_X = df.iloc[:, 1:]
final_model = KNeighborsClassifier(n_neighbors=4)
final_model.fit(X_train, Y_train)
final_pred = final_model.predict(final_X)

passengers = df.iloc[:, 0]
passengers = passengers.to_frame()
print(passengers)
passengers['Survived'] = final_pred
print(passengers)
passengers.to_csv('result.csv')
