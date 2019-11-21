import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from math import floor

#Converti un charactère en son nombre ascii
def toAscii(x):
	return ord(x)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
dataset = pandas.read_csv(url, names=names, na_values='?')

#Nous remplaçons les valeurs NAN de "stalk-root" par 'c' qui est la valeur apparaissant le plus
dataset = dataset.fillna('c');

#Nous convertissons les données qui sont des lettres en chiffre pour pouvoir appliquer l'algo
#Ce n'est pas la bonne méthode, il faut utiliser dummies de pandas pour transformer chaque valeur d'attribut en nouveaux attributs, mais ici cela ne changeait rien à nos résultats.
dataset = dataset.applymap(toAscii);

	
# Split-out validation dataset
array = dataset.values
X = array[:,1:23]
Y = array[:,0]
validation_size = 0.5
seed = 7
#partage le dataset en 2: 50% pour la validation et le reste pour le training
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

#SELECTION DU MEILLEUR MODELE

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=20, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Make predictions on validation dataset
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# On entre les nouvelles valeurs déterminant si l'Amanite tue-mouches est comestible ou non.
#xFlyAgaric = ['f','s','e','f','n','f','w','b','w','t','b','f','f','w','w','u','w','o','l','w','y','d']
xFlyAgaric = ['f','s','e','f','p','f','w','n','w','e','b','s','s','w','w','u','w','o','l','w','s','d']
var = list(map(toAscii,xFlyAgaric))
answer = cart.predict([var])
if chr(answer) == 'e':
	print("L'Amanite tue mouche est comestible")
else:
	print("L'Amanite tue mouche est vénéneuse")

sklearn.tree.export_graphviz(cart,out_file='tree.dot')























