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

#*Recibe siempre los sets de train y test
def logisticRegression(xTrain, xTest, yTrain, yTest):
    #*Se usa el logistic regression
    model = LogisticRegression(max_iter=10000)
    #*Se entrena el modelo
    model.fit(xTrain, yTrain)
    #*Se realizan las predicciones
    yOutput = model.predict(xTest)

    #*Se realizan las metricas de evaluacion
    accuracy = accuracy_score(y_test, yOutput)
    #*En estos casos se usa por el tipo de problema con multiples variables y por los resultados que sean una division por 0
    precision = precision_score(y_test, yOutput, average='weighted', zero_division=1)
    sensitivity = recall_score(y_test, yOutput, average='weighted', zero_division=1)
    #*Toma los primeros 4 valores de cada fila del resultado de la matriz para poder realizar la evaluacion
    tn, fp, fn, tp = confusion_matrix(y_test, yOutput).ravel()[:4]
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, yOutput, average='weighted')

    print("===== Logistic Regression =======")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"F1 Score: {f1}")

logisticRegression(X_train, X_test, y_train, y_test)