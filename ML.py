import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import statistics
from sklearn.model_selection import cross_val_score
from prefixspan import PrefixSpan
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

###Importation des données d'entrainement et de test:
train = pd.read_csv("/Users/thomasclement/Downloads/train.csv")
test = pd.read_csv("//Users/thomasclement/Downloads/test.csv")


### 1ère étape, afin de fournir à notre modèle des données de qualité, nous allons appliquer la méthode des 4C : Correcting, Completing, Creating, and Converting
#Correcting : Nous allons analyser le dataset afin d'identifier s'il existe des valeurs manquantes ou aberrantes pour chaque attributs
#Completing :Comme nous l'avons vu en TD les arbres de régressions n'aiment pas les valeurs manquantes/nuls. Nous avons donc plusieurs options : supprimer les lignes ou colonnes comprenant trop de valeurs manquantes ou les remplacer par estimation
#Creating : A partir des données que nous possédons nous pourrions peut-être créer d'autres variables intéressantes à exploiter dans notre modèle
#Converting : Certaines variables peuvent être dans des formats très différents et difficilement interprétables, nous pourrions donc les convertir afin de les rendre plus exploitables.

#nous créons un data frame comprenant les deux bases afin d'appliquer les changements aux deux 
def concat_df(train, test):
    # Returns a concatenated df of training and test set
    return pd.concat([train, test], sort=True).reset_index(drop=True)

df_all = concat_df(train,test)

print('Train columns with null values:\n', train.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', test.isnull().sum())
print("-"*10)

###Extraction et simplification des titres:
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
#Extraction des titres:
df_all['Title'] = df_all.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
#Simplification des titres: 
norm_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

df_all.Title = df_all.Title.map(norm_titles)
print(df_all.Title.value_counts())


df_all['Name_Len'] = df_all['Name'].apply(lambda x: len(x))
df_all['Survived'].groupby(pd.qcut(df_all['Name_Len'],5)).mean()

c = ['#24678d','#4caac9','#626262','#d5d5d5','#248d6c']
fig = plt.figure(figsize=(6,3.5), dpi=1600)
sns.kdeplot(df_all.Name_Len[(df_all.Survived==0)], 
            shade=True, alpha=0.5,color=c[2],label='Mort')
sns.kdeplot(df_all.Name_Len[(df_all.Survived==1)], 
            shade=True, alpha=0.5,color=c[1],label='Survivant')
plt.title('Survie en fonction de la taille du nom')
plt.legend(); plt.show()

###calculer les fréquences des tickets :
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

###Traitement des données pour fare
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
df_all['Fare'] = df_all['Fare'].fillna(med_fare)


###Trouver le nombre de personne dans une famille
df_all['Family'] = df_all['SibSp'] + df_all['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df_all['Family_Size_Grouped'] = df_all['Family'].map(family_map)


###Traitement des données sur les decks
#Grâce au schema du titanic nous pouvons supposer que certains decks étaient plus proches que d'autres des canaux de sauvatage
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
df_all['Deck'].value_counts()




### max et min par titre : vérifier si outlier
df_all.groupby(['Title','Pclass'])['Age'].agg(['min','max','count','mean'])


###Matrice de corrélation
#df_all.drop("PassengerId", axis = 1, inplace = True)

#f,ax = plt.subplots(figsize=(15, 13))
#sns.heatmap(train.corr(), annot=True, cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)
#plt.show()


###Traitement des données pour l'âge
#Nous avons 177 valeurs manquantes pour l'âge.
#L'âge en tant que tel n'est pas si déterminant pour le taux de survie mais il peut nous aider à catégoriser les individus :enfant, adultes, personnes âgés qui sont des variables que l'on peut supposer plus déterminantes

df_all.groupby(['Pclass','Family']).apply(lambda x: x.Age.isnull().sum()/len(x))
df_all.Title.value_counts()
clustering = df_all.groupby(['Sex','Pclass', 'Title'])
clustering.Age.median()

def newage (col):
    Sex=col[0]
    Pclass=col[1]
    Title=col[2]
    Age=col[3]
    if pd.isnull(Age):
        if Sex=="female" and Pclass==1 and Title=="Royalty":
            return 39.0
        elif Sex=="female" and Pclass==1 and Title=="Miss":
            return 30.0
        elif Sex=="female" and Pclass==1 and Title=="Officer": 
            return 49.0
        elif Sex=="female" and Pclass==1 and Title=="Mrs":
            return 45.0
        elif Sex=="female" and Pclass==2 and Title=="Miss":
            return 20.0
        elif Sex=="female" and Pclass==2 and Title=="Mrs":
            return 30.0
        elif Sex=="female" and Pclass==3 and Title=="Miss":
            return 18.0
        elif Sex=="female" and Pclass==3 and Title=="Mrs":
            return 31.0
        
        elif Sex=="male" and Pclass==1 and Title=="Officer":
            return 52.0
        elif Sex=="male" and Pclass==1 and Title=="Royalty":
            return 40
     
        elif Sex=="male" and Pclass==1 and Title=="Master":
            return 6.0
        elif Sex=="male" and Pclass==1 and Title=="Mr":
            return 41.5       
        elif Sex=="male" and Pclass==2 and Title=="Officer":
            return 41.5
        elif Sex=="male" and Pclass==2 and Title=="Master":
            return 2.0 
        elif Sex=="male" and Pclass==2 and Title=="Mr":
            return 30.0
        elif Sex=="male" and Pclass==3 and Title=="Master":
            return 6.0 
        else:
            return 26.0
    else:
        return Age
    
df_all.Age=df_all[['Sex','Pclass','Title','Age']].apply(newage, axis=1)
df_all.Age.value_counts()

df_all.groupby(['Sex','Pclass','Title','Family_Size_Grouped'])['Age'].agg(['min','max','count','median'])

cont_features = ['Age', 'Fare']
surv = df_all['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):    
    # Distribution of survival in feature
    sns.distplot(df_all[~surv][feature], label='Mort', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(df_all[surv][feature], label='Survivant', hist=True, color='#2ecc71', ax=axs[0][i])
    
    # Distribution of feature in dataset
    sns.distplot(df_all[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(df_all[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
    
    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution des survivants par {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)
        
plt.show()

#Catégorisation de la variable 'Fare' pour optimiser l'entrainement du modèle
df_all['Fare_ind']= df_all['Fare']/df_all['Ticket_Frequency']
df_all['Fare_ind'] = pd.qcut(df_all['Fare'], 5)
g3 = sns.factorplot(x="Fare_ind", y ="Survived", data=df_all, kind="bar", size=3)
g3.set(xlabel="Catégorie de prix")
plt.show()
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
###créer une variable catégorielle pour l'âge :
#df_all['age_grp'] = ""
#df_all.loc[ df_all['Age'] <= 16, 'age_grp'] 					     = '0-16'
#df_all.loc[(df_all['Age'] > 16) & (df_all['Age'] <= 32), 'age_grp'] = '16-32'
#df_all.loc[(df_all['Age'] > 32) & (df_all['Age'] <= 48), 'age_grp'] = '32-48'
#df_all.loc[(df_all['Age'] > 48) & (df_all['Age'] <= 64), 'age_grp'] = '48-64'
#df_all.loc[ df_all['Age'] > 64, 'age_grp']                          = '64+'
#df_all['age_grp'] = pd.to_numeric(df_all['age_grp'], errors='coerce')

df_all['age_grp'] = ""
df_all.loc[ df_all['Age'] <= 2, 'age_grp'] 					     = 0
df_all.loc[(df_all['Age'] > 2) & (df_all['Age'] <= 10), 'age_grp'] = 1
df_all.loc[(df_all['Age'] > 10) & (df_all['Age'] <= 19), 'age_grp'] = 2
df_all.loc[(df_all['Age'] > 19) & (df_all['Age'] <= 60), 'age_grp'] = 3
df_all.loc[ df_all['Age'] > 60, 'age_grp']                          = 4
df_all['age_grp'] = pd.to_numeric(df_all['age_grp'], errors='coerce')

g2 = sns.factorplot(x="age_grp", y ="Survived", data=df_all, kind="bar", size=3)
g2.set(xlabel="Catégorie d'âge")
plt.show()

###remplacer l'information du deck manquant par celle trouvé sur le forum Kaggle:
df_all['Embarked'] = df_all['Embarked'].fillna('S')

###traitement des valeurs pour cabines :
df_all['Cabin'] = df_all['Cabin'].fillna('Missing')
df_all['Cabin'] = df_all['Cabin'].str[0]


### Création de la variable : nombre d'enfants par ticket
youngot = df_all[df_all['age_grp'] == 0]
youngot.groupby('Ticket')['Ticket'].agg('count').to_frame('nbyticket').reset_index()

youngot = youngot.groupby('Ticket')['Ticket'].agg('count').to_frame('nbyticket').reset_index()

df_all = pd.merge(df_all, youngot, how='left', on="Ticket")
df_all['nbyticket'] = df_all['nbyticket'].fillna(0)

df_all.groupby(['nbyticket','Survived'])['nbyticket'].count()

g = sns.factorplot(x="nbyticket", y ="Survived", data=df_all, kind="bar", size=3)
g.set(xlabel="Nombre d'enfants par ticket")
plt.show()

df_adult=df_all[df_all['age_grp'] > 0]
g1 = sns.factorplot(x="nbyticket", y ="Survived", data=df_adult, kind="bar", size=3)
g1.set(xlabel="Nombre d'enfants par ticket")
plt.show()

### règles d'association ###
#Importation des données
train= df_all[:891]
train = train.drop(['Ticket','Cabin','Name','Fare','SibSp','Parch'],axis=1).set_index('PassengerId')
train.Survived[train.Survived==1.0] = "Survived"
train.Survived[train.Survived==0.0] = "Dead"
train.Age = train.Age.astype(int)
train = train[['Survived','Pclass','age_grp','Title','Embarked','Family_Size_Grouped']]
train = train.astype(str).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(train).transform(train)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

#Frequence des caractéristiques trouvés présents à au moins 30% dans la dataset
freqitems = fpgrowth(df, min_support = 0.3, use_colnames = True)
print(freqitems)

rules = association_rules(freqitems, metric="confidence",min_threshold=0.6)
print(rules)


subresult = rules[["antecedents", "consequents", "support","confidence"]]
subresult.to_csv('Association Rules Projet.csv', sep='\t', encoding='utf-8')

print(subresult.head())

###encoding des variables:
non_numeric_features = ['Embarked', 'Sex','Cabin', 'Deck', 'Title', 'Family_Size_Grouped', 'Fare_ind','Fare']

for feature in non_numeric_features:        
    df_all[feature] = LabelEncoder().fit_transform(df_all[feature])


cat_features = ['Pclass', 'Sex','Title', 'Embarked', 'Family_Size_Grouped','age_grp','Deck','Cabin','nbyticket']
encoded_features = []

for feature in cat_features:
    encoded_feat = OneHotEncoder().fit_transform(df_all[feature].values.reshape(-1, 1)).toarray()
    n = df_all[feature].nunique()
    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded_feat, columns=cols)
    encoded_df.index = df_all.index
    encoded_features.append(encoded_df)



df_all = pd.concat([df_all, *encoded_features[:9]], axis=1)

###Création des dataframe comprenant les features que nous allons utiliser et du dataframe de la variable de classification:
df_x = df_all.drop(['Age','Embarked','Name','Parch','SibSp','PassengerId','Pclass','Ticket','Sex','Survived','Title','Family','Family_Size_Grouped','Deck','Ticket_Frequency','age_grp','Cabin','nbyticket','Fare_ind'], axis=1)
df_y = df_all['Survived']

#Division des datatsets de test et d'entrainement comme à leur origine
x_train = df_x.loc[:890]
x_test = df_x.loc[891:]
y_train = df_y.loc[:890]
y_train=y_train.astype('int')

######### création des modèles ##########
###Random Forest classifier
rf = RandomForestClassifier(random_state = 2)
cv = cross_val_score(rf,x_train,y_train,cv=10)
print(cv)
print(cv.mean())

###SVM classifier
svc = SVC(probability = True)
cv2 = cross_val_score(svc,x_train,y_train,cv=10)
print(cv2)
print(cv2.mean())


### choix des hyperparamètres de nos modèles avec GridSearchCV
# fonction pour le report de performance des modèles
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
  
    
##############################################################################
##Pour le SVM :
svc1 = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
clf_svc = GridSearchCV(svc1, param_grid = param_grid, cv = 5, verbose = True)
best_clf_svc = clf_svc.fit(x_train,y_train)
clf_performance(best_clf_svc,'SVC')
best_svc = best_clf_svc.best_estimator_

##############################################################################
##Pour le RF :
rf1 = RandomForestClassifier(random_state = 2)
param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [15, 20, 25],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2,3],
                                  'min_samples_split': [2,3]}
                                  
clf_rf = GridSearchCV(rf1, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(x_train,y_train)
clf_performance(best_clf_rf,'Random Forest')
best_rf = best_clf_rf.best_estimator_

best_rf.fit(x_train,y_train)
y_best_rf = best_rf.predict(x_test).astype('int')
results_best_rf = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_best_rf
    })
results_best_rf.to_csv('result_best_rf.csv', index=False)

feat_importances2 = pd.Series(best_rf.feature_importances_, index=x_train.columns)
feat_importances2.nlargest(20).plot(kind='barh')


##############################################################################
# POUR ALLER PLUS LOIN : BOOSTING ET SYSTEME DE VOTE
##############################################################################

###Régression logistique :
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(x_train,y_train)
best_lr = best_clf_lr.best_estimator_

###XGBOOST :
from xgboost import XGBClassifier
XGB = XGBClassifier(random_state = 2)

xgb_param_grid = {'learning_rate': [0.1,0.04,0.01], 
                  'max_depth': [5,6,7],
                  'n_estimators': [350,400,450,2000], 
                  'gamma': [0,1,5,8],
                  'subsample': [0.8,0.95,1.0]}

gsXBC = GridSearchCV(XGB, param_grid = xgb_param_grid, cv = 5, scoring = "accuracy", n_jobs = -1, verbose = True)

gsXBC.fit(x_train,y_train)

best_xgb = gsXBC.best_estimator_
best_xgb.fit(x_train,y_train)

y_best_xgb = best_xgb.predict(x_test).astype('int')
results_best_xgb = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_best_xgb
    })
results_best_xgb.to_csv('result_best_xgb.csv', index=False)



###Vote :
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('rf', best_rf), 
                                      ('lr', best_lr),
                                      ('svc', best_svc),
                                      ('xgb', best_xgb)])

v_param_grid = {'voting':['soft',
                          'hard']} # tuning voting parameter

gsV = GridSearchCV(voting, 
                   param_grid = 
                   v_param_grid, 
                   cv = 5, 
                   scoring = "accuracy",
                   n_jobs = -1, 
                   verbose = True)

gsV.fit(x_train,y_train)

v_best = gsV.best_estimator_
y_v_best = v_best.predict(x_test).astype('int')
results_v_best = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_v_best
    })
results_v_best.to_csv('result_v_best.csv', index=False)
