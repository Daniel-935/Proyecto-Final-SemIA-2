import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#*Para las metricas de evaluacion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

#*Establecemos los nombres de las columnas o nombres de variables para poder leer y hacer la clasificacion del archivo csv
variableNames = ['animal_name', 
    'hair', 'feathers', 
    'eggs', 'milk', 
    'airborne', 'aquatic',
    'predator', 'toothed',
    'backbone', 'breathes',
    'venomous', 'fins',
    'legs', 'tail',
    'domestic', 'catsize',
    'class_type'
]

data = pd.read_csv('zoo.csv', header=None, names=variableNames)
print(data)
#*Se divide el dataset en X y Y (entradas / salidas)
X = data.drop(['animal_name', 'class_type'], axis=1)
y = data['class_type']

#*Se crean los sets para train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
