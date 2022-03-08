# %% [markdown]
# # HOJA DE TRABAJO 3 ARBOLES DE DECISION

# Raul Jimenez 19017

# Donaldo Garcia 19683

# Oscar Saravia 19322

# %%
# from re import U
from statsmodels.graphics.gofplots import qqplot
import numpy as np
import pandas as pd
# import pandasql as ps
import matplotlib.pyplot as plt
# import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
# import statsmodels.api as sm
import seaborn as sns
# import random
import sklearn.cluster as cluster
# import sklearn.metrics as metrics
import sklearn.preprocessing
# import scipy.cluster.hierarchy as sch
import pyclustertend
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics
# import sklearn.mixture as mixture
# from sklearn import datasets
# from sklearn.cluster import DBSCAN
# from numpy import unique
# from numpy import where
# from matplotlib import pyplot
# from sklearn.datasets import make_classification
# from sklearn.cluster import Birch
# from sklearn.mixture import GaussianMixture

from sklearn.model_selection import train_test_split

# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# %% [markdown]

# ## 1. Descargue los conjuntos de datos de la plataforma kaggle.

# %%
train = pd.read_csv('./train.csv', encoding='latin1')
train.head()

# %% [markdown]

# ## 2. Haga  un  análisis  exploratorio  extenso  de  los  datos.  Explique  bien  todos  los  hallazgos.  No ponga solo gráficas y código. Debe llegar a conclusiones interesantes para poder predecir.Explique el preprocesamiento que necesitó hacer.

# Se deciden utilizar estas variables debido a que estas son las que nos permiten predecir el comportamiento de este mercad o en un futoro. Con estas variables podemos ver si tiene alguna importanacia en el precio la cantidad del espacio, cantidad de cuartos/baños e incluso el año en el que se termina vendiendo la casa


# %% [markdown]
# - SalePrice - **CUANTITATIVO CONTINUO** debido a que el precio puede tener centavos; the property's sale price in dollars. This is the target variable that you're trying to predict.
# - LotArea: **CUANTITATIVO CONTINUO** Lot size in square feet
# - OverallCond: **CUANTITATIVO DISCRETO** Overall condition rating
# - YearBuilt: **CUANTITATIVO DISCRETO** Original construction date
# - MasVnrArea: **CUANTITATIVO CONTINUO** Masonry veneer area in square feet
# - TotalBsmtSF: **CUANTITATIVO CONTINUO** Total square feet of basement area
# - 1stFlrSF: **CUANTITATIVO CONTINUO** First Floor square feet
# - 2ndFlrSF: **CUANTITATIVO CONTINUO** Second floor square feet
# - GrLivArea: **CUANTITATIVO CONTINUO** Above grade (ground) living area square feet
# - TotRmsAbvGrd: **CUANTITATIVO DISCRETO** Total rooms above grade (does not include bathrooms)
# - GarageCars: **CUANTITATIVO DISCRETO** Size of garage in car capacity
# - WoodDeckSF: **CUANTITATIVO CONTINUO** Wood deck area in square feet
# - OpenPorchSF: **CUANTITATIVO CONTINUO** Open porch area in square feet
# - EnclosedPorch: **CUANTITATIVO CONTINUO** Enclosed porch area in square feet
# - PoolArea: **CUANTITATIVO CONTINUO** Pool area in square feet
# - Neighborhood: **CUALITATIVO NOMINAL** Physical locations within Ames city limits

# %%
usefullAttr = ['SalePrice', 'LotArea', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF',
               '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'PoolArea', 'Neighborhood']


# %%
data = train[usefullAttr]
data.head()

# %% [markdown]
# ### GRAFICAS DE VARIABLES

# %%


def get_histogram_qq(variable):
    plt.hist(x=data[variable] .dropna(), color='#F2AB6D', rwidth=1)
    plt.title(f'Histograma de la variable{variable}')
    plt.xlabel(variable)
    plt.ylabel('frencuencias')
    plt.rcParams['figure.figsize'] = (30, 30)
    plt.show()

    distribucion_generada = data[variable].dropna()
    # Represento el Q-Q plot
    qqplot(distribucion_generada, line='s')
    plt.show()

# %% [markdown]
# #### SalePricee
# Se puede determinar que la variable SalePrice no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.


# %%
get_histogram_qq('SalePrice')

# %% [markdown]
# #### LotArea
# Se puede determinar que la variable LotArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('LotArea')

# %% [markdown]
# #### OverallCond
# Se puede determinar que la variable OverallCond no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('OverallCond')

# %% [markdown]
# #### YearBuilt
# Se puede determinar que la variable YearBuilt no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('YearBuilt')

# %% [markdown]
# #### MasVnrArea
# Se puede determinar que la variable MasVnrArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('MasVnrArea')

# %% [markdown]
# #### TotalBsmtSF
# Se puede determinar que la variable TotalBsmtSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('TotalBsmtSF')

# %% [markdown]
# #### 1stFlrSF
# Se puede determinar que la variable 1stFlrSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('1stFlrSF')

# %% [markdown]
# #### 2ndFlrSF
# Se puede determinar que la variable 2ndFlrSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('2ndFlrSF')

# %% [markdown]
# #### GrLivArea
# Se puede determinar que la variable GrLivArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('GrLivArea')

# %% [markdown]
# #### TotRmsAbvGrd
# Se puede determinar que la variable TotRmsAbvGrd no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('TotRmsAbvGrd')

# %% [markdown]
# #### GarageCars
# Se puede determinar que la variable GarageCars no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('GarageCars')

# %% [markdown]
# #### WoodDeckSF
# Se puede determinar que la variable WoodDeckSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('WoodDeckSF')

# %% [markdown]
# #### OpenPorchSF
# Se puede determinar que la variable OpenPorchSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('OpenPorchSF')

# %% [markdown]
# #### EnclosedPorch
# Se puede determinar que la variable EnclosedPorch no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('EnclosedPorch')

# %% [markdown]
# #### PoolArea
# Se puede determinar que la variable PoolArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('PoolArea')

# %% [markdown]
# #### Neighborhood
# Se puede determinar que la variable Neighborhood no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
eje_x = np.array(pd.value_counts(data['Neighborhood']).keys())
eje_y = pd.value_counts(data['Neighborhood'])

plt.bar(eje_x, eje_y)
plt.rcParams['figure.figsize'] = (10, 10)
plt.ylabel('Frecuencia de la variable Neighborhood')
plt.xlabel('Años')
plt.title('Grafico de barras para la variable Neighborhood')
plt.show()

# %% [markdown]
# ## 3. Incluya un análisis de grupos en el análisis exploratorio. Explique las características de los grupos.
# Se puede concluir que los datos normalizados son viables para el uso de clusters o grupos. Se logra llegar a esta conclucion debido a que nuestro test de hopkins sale de 0.08 junto con la grafica VAT. Con la grafica del codo se puede determinar que se pueden utilizar dos clusters debido a que es en ese dato donde se encuentra mas marcado el codo. Pero tambien se podria usar 7 debido a que tambien se encuenatra marcada ahi una punta.

# %%
data.hist()
plt.show()

# %%
# NORMALIZAMOS DATOS
if 'Neighborhood' in data.columns:
    usefullAttr.remove('Neighborhood')
data = train[usefullAttr]
X = []
for column in data.columns:
    try:
        column
        if column != 'Neighborhood' or column != 'SalePrice':
            data[column] = (data[column]-data[column].mean()) / \
                data[column].std()
            X.append(data[column])
    except:
        continue
data_clean = data.dropna(subset=usefullAttr, inplace=True)
X_Scale = np.array(data)
X_Scale

# %%
# HOPKINGS
X_scale = sklearn.preprocessing.scale(X_Scale)
# X = X_scale
pyclustertend.hopkins(X_scale, len(X_scale))

# %%
# VAT
pyclustertend.vat(X_Scale)

# devolvemos el SalePrice a su valor original
data['SalePrice'] = train['SalePrice']

# %%
numeroClusters = range(1, 11)
wcss = []
for i in numeroClusters:
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(X_Scale)
    wcss.append(kmeans.inertia_)

plt.plot(numeroClusters, wcss)
plt.xlabel("Número de clusters")
plt.ylabel("Score")
plt.title("Gráfico de Codo")
plt.show()

# %%
kmeans = cluster.KMeans(n_clusters=7)
kmeans.fit(X_Scale)
kmeans_result = kmeans.predict(X_Scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = np.where(kmeans_result == kmeans_cluster)
    # make the plot
    plt.scatter(X_Scale[index, 0], X_Scale[index, 1])
plt.show()

# %%
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X_Scale)
kmeans_result = kmeans.predict(X_Scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = np.where(kmeans_result == kmeans_cluster)
    # make the plot
    plt.scatter(X_Scale[index, 0], X_Scale[index, 1])
plt.show()

# %%
data['cluster'] = kmeans.labels_
print(data[data['cluster'] == 0].describe().transpose())
print(data[data['cluster'] == 1].describe().transpose())
print(data[data['cluster'] == 2].describe().transpose())

# %% [markdown]
# ## 4. Dependiendo del análisis exploratorio elaborado cree una variable respuesta que le permita clasificar  las  casas  en  Económicas,  Intermedias  o  Caras.  Los  límites  de  estas  clases  deben tener un fundamento en la distribución de los datos de precios, y estar bien explicados.

# Se crea la variable de 'Clasificacion' en la cual se clasifica como Economica, Intermedia o Cara. Acorde al precio. Para obtener el rango se crean dos limites. El limite 1 es la media de nuestro salesprice que tienen cluster 0 y para el limit 2 es la media de los que tienen cluster 1. Se realizo de esta forma debido a que como muchos de los precios tenian maximos y minimos en dos categoris se uso la media para colocar los limites.

# %%
# Clasificacion de casas en: Economias, Intermedias o Caras.
data.fillna(0)
limit1 = data.query('cluster == 0')['SalePrice'].mean()
limit2 = data.query('cluster == 1')['SalePrice'].mean()
# minPrice = data['SalePrice'].min()
# maxPrice = data['SalePrice'].max()
# division = (maxPrice - minPrice) / 3
data['Clasificacion'] = data['LotArea']

# data['Clasificacion'][data['SalePrice'] < minPrice + division] = 'Economica'
# data['Clasificacion'][data['SalePrice'] >= minPrice + division] = 'Intermedia'
# data['Clasificacion'][data['SalePrice'] >= minPrice + division * 2] = 'Caras'
data.loc[data['SalePrice'] < limit1, 'Clasificacion'] = 'Economica'
data.loc[(data['SalePrice'] >= limit1) & (
    data['SalePrice'] < limit2), 'Clasificacion'] = 'Intermedia'
data.loc[data['SalePrice'] >= limit2, 'Clasificacion'] = 'Caras'
# data.loc[data['cluster'] == 0, 'Clasificacion'] = 'Economica'
# data.loc[data['cluster'] == 1, 'Clasificacion'] = 'Intermedia'
# data.loc[data['cluster'] == 2, 'Clasificacion'] = 'Caras'

# %% [markdown]
# #### Contamos la cantidad de casas por clasificacion

# %%
# Obtener cuantos datos hay por cada clasificacion
print(data['Clasificacion'].value_counts())

# %% [markdown]
# ## 5. Divida el set de datos preprocesados en dos conjuntos: Entrenamiento y prueba. Describa el criterio que usó para crear los conjuntos: número de filas de cada uno, estratificado o no, balanceado o no, etc. Si le proveen un conjunto de datos de prueba y tiene suficientes datos, tómelo como de validación, pero haga sus propios conjuntos de prueba.

# Se dividen los en dos grupos, entranamiento con 0.7 y prueba con 0.3. Se utiliza el caso normal debido a que evitamos sobreajuste de nuestros datos. Y va a ser nuestra variable clasificacion, X con los demas datos disponibles sin la clasificacion. Se estratifican los datos debdio a que tenemos 3 tipos de clasiicacion Economia, Intermedia y Cara.

# %% [markdown]
# # Estableciendo los conjuntos de Entrenamiento y Prueba

# %%
y = data['Clasificacion']
X = data.drop(['Clasificacion', 'SalePrice'], axis=1)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7)
y_train

# %% [markdown]
# 70% de entrenamiento y 30% prueba

# %%
arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol = arbol.fit(X_train, y_train)

# %% [markdown]
# ## 6. Elabore  el  árbol  de  clasificación  utilizando  el  conjunto  de  entrenamiento  y  la  variable respuesta que creó en el punto 4.  Explique los resultados a los que llega. Muestre el modelo gráficamente. El experimento debe ser reproducible por lo que debe fijar que los conjuntos de entrenamiento y prueba sean los mismos siempre que se ejecute el código.

# %%
tree.plot_tree(arbol, feature_names=data.columns,
               class_names=['0', '1', '2'], filled=True)
# %%
y_pred = arbol.predict(X_test)
print(y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(
    y_test, y_pred, average='weighted'))
print("Recall: ", metrics.recall_score(y_test, y_pred, average='weighted'))

# %% [markdown]
# ## 7.Elabore el árbol de regresión para predecir el precio de las viviendas utilizando el conjunto
# de entrenamiento.  Explique los resultados a los que llega. Muestre el modelo gráficamente.
# El experimento debe ser reproducible por lo que debe fijar que los conjuntos de
# entrenamiento y prueba sean los mismos siempre que se ejecute el código.
plt.rcParams['figure.figsize'] = (160, 90)
regressionTree = DecisionTreeRegressor(max_depth=4, random_state=42)
regressionTree = arbol.fit(X_train, y_train)
tree.plot_tree(regressionTree, feature_names=data.columns,
               class_names=['0', '1', '2'], filled=True)

# %%[markdown]
# ## 8.Utilice  el  modelo  con  el  conjunto  de  prueba  y  determine  la  eficiencia  del  algoritmo  para clasificar y predecir, en dependencia de las características de la variable respuesta.

y_pred = arbol.predict(X_test)
print(y_pred)
print("Exactitud:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(
    y_test, y_pred, average='weighted'))
print("Recall: ", metrics.recall_score(y_test, y_pred, average='weighted'))

# %%[markdown]
# ## 9.Haga un análisis de la eficiencia del algoritmo usando una matriz de confusión para el árbol de clasificación. Tenga en cuenta la efectividad, donde el algoritmo se equivocó más, donde se equivocó menos y la importancia que tienen los errores
confussion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confussion_matrix)
sns.heatmap(confussion_matrix, annot=True)

# %%[markdown]
# ## 10.Analice el desempeño del árbol de regresión.
tree.plot_tree(regressionTree, feature_names=data.columns,
               class_names=['0', '1', '2'], filled=True)

# %%
