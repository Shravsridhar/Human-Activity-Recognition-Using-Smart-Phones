import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import confusion_matrix
import time
import scikitplot as skplt

# load files and convert into matrix format
def file_load(path):
	df = pd.read_csv(path)
	return df

# load train dataset and labels

df_train = file_load("C:\\Users\\arroh\\Desktop\\Data Mining\\Choun Chou\\Project\\train.csv")
df_test =  file_load("C:\\Users\\arroh\\Desktop\\Data Mining\\Choun Chou\\Project\\test.csv")

df_train.head()
df_test.head()

# Check null values
df_train.isnull().values.any()
df_test.isnull().values.any()

# No null values in train and test data


# Subject col not usefull - drop
df_train.drop('subject', axis =1, inplace=True)
df_test.drop('subject', axis =1, inplace=True)

# explore
STANDING = df_train[df_train['Activity']=="STANDING"]
LAYING =  df_train[df_train['Activity']=="LAYING"]
SITTING =  df_train[df_train['Activity']=="SITTING"]
WALKING =  df_train[df_train['Activity']=="WALKING"]
WALKING_UPSTAIRS =  df_train[df_train['Activity']=="WALKING_UPSTAIRS"]
WALKING_DOWNSTAIRS =  df_train[df_train['Activity']=="WALKING_DOWNSTAIRS"]


# Encoding target - converting non-num to num variable
le = preprocessing.LabelEncoder()
for x in [df_train,df_test]:
    x['Activity'] = le.fit_transform(x.Activity)

# Split into features and class
df_traindata, df_trainlabel = df_train.iloc[:, 0:len(df_train.columns) - 1], df_train.iloc[:, -1]
df_testdata, df_testlabel = df_test.iloc[:, 0:len(df_test.columns) -1], df_test.iloc[:, -1]

# understanding features
# Train data
feat_train = pd.DataFrame.from_dict(Counter([col.split('-')[0].split('(')[0] for col in df_traindata.columns]), orient='index').rename(columns={0:'count'}).sort_values('count', ascending=False)
# Test data
feat_test = pd.DataFrame.from_dict(Counter([col.split('-')[0].split('(')[0] for col in df_testdata.columns]), orient='index').rename(columns={0:'count'}).sort_values('count', ascending=False)


classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(),
    GaussianNB(),
    LogisticRegression()
    ]


# Naive Train Accuracy
algo = []
scores = []
for clf in classifiers:
    algo.append(clf.__class__.__name__)
    scores.append(cross_val_score(clf,df_traindata,df_trainlabel, cv=5).mean())
Naivescore_df_Train = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')

Naivescore_df_Train


# Naive Test Accuracy

algo = []
scores = []

for clf in classifiers:
    clf = clf.fit(df_traindata, df_trainlabel)
    y_pred = clf.predict(df_testdata)
    algo.append(clf.__class__.__name__)
    scores.append(accuracy_score(y_pred, df_testlabel))
Naivescore_df_Test  = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')
Naivescore_df_Test

Naive_Comp=pd.merge(Naivescore_df_Test,Naivescore_df_Train,on="Algorithm")


# Plot Accuracy
a1 = Naivescore_df_Train.plot.bar( color=(0.2, 0.4, 0.6, 0.5))
a1.set_xticklabels(Naivescore_df_Train.index, rotation = 40, fontsize=7)

fig = plt.figure(figsize=(5,5)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .3

Naivescore_df_Train.Score.plot(kind='bar',color='green',ax=ax,width=width, position=0)
Naivescore_df_Test.Score.plot(kind='bar',color='red', ax=ax2,width = width,position=1)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Train')
ax2.set_ylabel('Test')

ax.set_xlim(-1,7)
plt.show()
plt.savefig('pltacc.png')

# Feature selection

# Extra Tree based feature selection
tree_clf = ExtraTreesClassifier()
tree_clf = tree_clf.fit(df_traindata,df_trainlabel)
model = SelectFromModel(tree_clf, prefit=True)
nf_tree_features_Train= df_traindata.loc[:, model.get_support()]
nf_tree_features_Test = df_testdata.loc[:, model.get_support()]
print(nf_tree_features_Train.shape)
# 92 features

# Train Score after Extra Tree Classifier

algo = []
scores = []

for clf in classifiers:
    algo.append(clf.__class__.__name__)
    scores.append(cross_val_score(clf,nf_tree_features_Train,df_trainlabel, cv=5).mean())
Extra_Train_score_df = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')
Extra_Train_score_df


# Test Score after Extra Tree Classifier
algo = []
scores = []

for clf in classifiers:
    clf = clf.fit(nf_tree_features_Train , df_trainlabel)
    y_pred = clf.predict(nf_tree_features_Test)
    algo.append(clf.__class__.__name__)
    scores.append(accuracy_score(y_pred, df_testlabel))
Extra_Test_score_df = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')
Extra_Test_score_df



Extra_Comp=pd.merge(Extra_Test_score_df,Extra_Train_score_df,on="Algorithm")


# Feature Importance
# Random Forest check
# Bagged decision trees for feature importance- embedded method
Rtree_clf = RandomForestClassifier()
Rtree_clf = Rtree_clf.fit(df_traindata,df_trainlabel)
model = SelectFromModel(Rtree_clf, prefit=True)
RF_tree_featuresTrain=df_traindata.loc[:, model.get_support()]
RF_tree_featuresTest = df_testdata.loc[:, model.get_support()]
print(RF_tree_featuresTrain.shape)
# 87 features

# print feature and its importance
importances = Rtree_clf.feature_importances_
names = RF_tree_featuresTrain.columns
a = sorted(zip(map(lambda x: round(x, 5), importances), names), reverse=True)
a = pd.DataFrame(a)
print(a)

# Train score for Random Forest
algo = []
scores = []

for clf in classifiers:
    algo.append(clf.__class__.__name__)
    scores.append(cross_val_score(clf,RF_tree_featuresTrain,df_trainlabel, cv=5).mean())
RFtree_Trainscore_df = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')
RFtree_Trainscore_df

# Compare learning curve and cv score
skplt.estimators.plot_learning_curve(Rtree_clf,RF_tree_featuresTrain,df_trainlabel)
# Test score for Random Forest
algo = []
scores = []

for clf in classifiers:
    clf = clf.fit(RF_tree_featuresTrain, df_trainlabel)
    y_pred = clf.predict(RF_tree_featuresTest)
    algo.append(clf.__class__.__name__)
    scores.append(accuracy_score(y_pred, df_testlabel))
RFtree_Testscore_df = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')
RFtree_Testscore_df

RFtree_Comp=pd.merge(RFtree_Testscore_df,RFtree_Trainscore_df,on="Algorithm")

# Decision Tree Check
DTtree_clf = DecisionTreeClassifier(random_state= 0)
DTtree_clf = DTtree_clf.fit(df_traindata,df_trainlabel)
model = SelectFromModel(DTtree_clf, prefit=True)
DT_tree_featuresTrain= df_traindata.loc[:, model.get_support()]
DT_tree_featuresTest= df_testdata.loc[:, model.get_support()]
print(DT_tree_featuresTrain.shape)
# 32 features

# train score for Decision Tree
algo = []
scores = []

for clf in classifiers:
    algo.append(clf.__class__.__name__)
    scores.append(cross_val_score(clf,RF_tree_featuresTrain,df_trainlabel, cv=5).mean())
DT_treeTrain_score_df = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')
DT_treeTrain_score_df

# test score for Decision Tree
algo = []
scores = []

for clf in classifiers:
    clf = clf.fit(DT_tree_featuresTrain, df_trainlabel)
    y_pred = clf.predict(DT_tree_featuresTest)
    algo.append(clf.__class__.__name__)
    scores.append(accuracy_score(y_pred, df_testlabel))
DT_treeTest_score_df = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')
DT_treeTest_score_df


DTtree_Comp=pd.merge(DT_treeTest_score_df,DT_treeTrain_score_df,on="Algorithm")



# Random Forest Feature Importance
for name, importance in zip(df_traindata, Rtree_clf.feature_importances_):
     print(name, "=", importance)
importances = Rtree_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in Rtree_clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
indices.shape
indices = indices[:200]
# Prints feature ranking
print("Feature ranking:")
for f in range(200):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plots feature importances

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.xlabel("# of Features ")
plt.ylabel("Importance Score")
plt.bar(range(200), importances[indices],color="r", yerr=std[indices], align="center")
plt.xlim([0, 200])
plt.show()
plt.show()
# Another method of plot
plt.figure()
plt.title("Feature importances")
plt.barh(range(df_traindata.shape[1]), importances[indices],color="r", xerr=std[indices], align="center")




# Applying RFE Cross validation to find number of features
# The "accuracy" scoring is proportional to the number of correct classifications

svc=SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfetrain=rfecv.fit(RF_tree_featuresTrain, df_trainlabel)
print('Optimal number of features :', rfecv.n_features_)
# 63
print('Best features :', RF_tree_featuresTrain.columns[rfecv.support_])


# Plot showing the Cross Validation score
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
# Applying RFE with optimal number of features
rfe = RFE(estimator=svc, n_features_to_select=39, step=1)
rfe = rfe.fit(RF_tree_featuresTrain, df_trainlabel)

rfe_train=RF_tree_featuresTrain.loc[:, rfe.get_support()]
rfe_test=RF_tree_featuresTest.loc[:, rfe.get_support()]



# Checking accuracy
# Train accuracy
cross_val_score(svc,rfe_train,df_trainlabel, cv=5).mean()
# Test Accuracy
scv = svc.fit(rfe_train, df_trainlabel)
y_pred = scv.predict(rfe_test)
accuracy_score(y_pred, df_testlabel)

# Variance threshold
selector = VarianceThreshold(0.95*(1-.95))
varsel=selector.fit(rfe_train)
rfe_train.loc[:, varsel.get_support()].shape
# 55
vartrain=rfe_train.loc[:, varsel.get_support()]
vartest=rfe_test.loc[:, varsel.get_support()]

cross_val_score(svc,vartrain,df_trainlabel, cv=5).mean()

# Test Accuracy
scv = svc.fit(vartrain, df_trainlabel)
y_pred = scv.predict(vartest)
accuracy_score(y_pred, df_testlabel)

# PCA
pca = PCA(n_components = 45)
pca_traindata = pca.fit(vartrain)

pca_traindata.explained_variance_
pca_traindata.n_components_
pcatrain = pca_traindata.transform(vartrain)
pcatest = pca_traindata.transform(vartest)
cum_ratio = (np.cumsum(pca_traindata.explained_variance_ratio_))


# Visualize PCA result
plt.plot(np.cumsum(pca_traindata.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# 21 features - constant after that
pca = PCA(n_components = 21)
pca_traindata = pca.fit(vartrain)

pca_traindata.explained_variance_
pca_traindata.n_components_
pcatrain = pca_traindata.transform(vartrain)
pcatest = pca_traindata.transform(vartest)
(np.cumsum(pca_traindata.explained_variance_ratio_))

# PCA in 2D projection
skplt.decomposition.plot_pca_2d_projection(pca, vartrain, df_trainlabel)


# Accuracy
cross_val_score(svc,pcatrain,df_trainlabel, cv=5).mean()
# Test Accuracy
scv = svc.fit(pcatrain, df_trainlabel)
y_pred = scv.predict(pcatest)
ac_score = accuracy_score(y_pred, df_testlabel)

df_pcacomp = pd.DataFrame(pca.components_,columns=vartrain.columns,index = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21'])
print (df_pcacomp)

cf_mat = confusion_matrix(df_testlabel, y_pred)
print("Accuracy: %f" %ac_score)
activities = le.classes_
plot_confusion_matrix(cf_mat, classes=activities,title="Confusion Matrix for Test data")


final_feat_train = pd.DataFrame.from_dict(Counter([col.split('-')[0].split('(')[0] for col in vartrain.columns]), orient='index').rename(columns={0:'count'}).sort_values('count', ascending=False)

LAYING = df_trainlabel[df_trainlabel==0]
SITTING = df_trainlabel[df_trainlabel==1]
STANDING = df_trainlabel[df_trainlabel==2]
WALKING =df_trainlabel[df_trainlabel==3]
WALKING_DOWNSTAIRS = df_trainlabel[df_trainlabel==4]
WALKING_UPSTAIRS = df_trainlabel[df_trainlabel==5]


fexp_df = pd.concat([vartrain, df_trainlabel], axis=1).reset_index(drop=True)
features = vartrain.columns
fexp_df.shape
importances = Rtree_clf.feature_importances_

data = {'Gyroscope':0, 'Accelerometer':0}
for importance, feature in zip(importances, features):
    if 'Gyro' in feature:
        data['Gyroscope'] += importance
    if 'Acc' in feature:
        data['Accelerometer'] += importance

# Create dataframe and plot
sensor_df = pd.DataFrame.from_dict(data, orient='index').rename(columns={0:'Importance'})
sensor_df.plot(kind='barh', figsize=(10,7), title='Sensor Importance by Feature Importance Sum')
plt.show()

f_ex = [ 'fBodyAccJerk-entropy()-X','angle(X,gravityMean)','tGravityAcc-arCoeff()-Y,2','tBodyAcc-correlation()-Y,Z',
'tBodyGyro-correlation()-Y,Z','tGravityAcc-sma()','fBodyAcc-kurtosis()-Y',
'tBodyAcc-entropy()-X','tBodyAcc-correlation()-Y,Z','tBodyAcc-correlation()-X,Y',
'tGravityAcc-arCoeff()-Y,3','fBodyAccJerk-maxInds-X','fBodyAccJerk-maxInds-X','fBodyGyro-max()-Y',
'fBodyGyro-std()-X','fBodyAccJerk-maxInds-X','tGravityAcc-entropy()-Y','fBodyGyro-maxInds-Z',
'tBodyGyroJerk-arCoeff()-X,1','tBodyGyroJerk-arCoeff()-X,1','tGravityAcc-entropy()-Y' ]


expl = vartrain.filter(items= f_ex)

Random
svc_fit = svc.fit(expl,df_trainlabel)
model = SelectFromModel(svc_fit, prefit=True)
svc_final_tr=df_traindata.loc[:, model.get_support()]
svc_final_te = df_testdata.loc[:, model.get_support()]
print(RF_tree_featuresTrain.shape)
# 87 features

# print feature and its importance
importances = model.feature_importances_
names = RF_tree_featuresTrain.columns
a = sorted(zip(map(lambda x: round(x, 5), importances), names), reverse=True)
a = pd.DataFrame(a)
print(a)



Rtree_clf = RandomForestClassifier()
Rtree_clf = Rtree_clf.fit(expl,df_trainlabel)
model = SelectFromModel(Rtree_clf, prefit=True)
RF_tree_featuresTrain=df_traindata.loc[:, model.get_support()]
RF_tree_featuresTest = df_testdata.loc[:, model.get_support()]
print(RF_tree_featuresTrain.shape)
# 87 features

# print feature and its importance
importances = Rtree_clf.feature_importances_

features = expl.columns

data = {'Gyroscope':0, 'Accelerometer':0}
for importance, feature in zip(importances, features):
    if 'Gyro' in feature:
        data['Gyroscope'] += importance
    if 'Acc' in feature:
        data['Accelerometer'] += importance
fexp_df.iloc[:,[57]]

# Create dataframe and plot
sensor_df = pd.DataFrame.from_dict(data, orient='index').rename(columns={0:'Importance'})
sensor_df.plot(kind='barh', figsize=(10,7), title='Sensor Importance by Feature Importance Sum')
plt.show()

for name, importance in zip(expl, Rtree_clf.feature_importances_):
     print(name, "=", importance)
importances = Rtree_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in Rtree_clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
indices.shape
indices = indices[:200]
# Prints feature ranking
print("Feature ranking:")
for f in range(200):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plots feature importances

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.xlabel("Features ")
plt.ylabel("Importance Score")
plt.bar(range(expl.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xlim([0, 30])
plt.xticks(range(expl.shape[1]), expl.columns[indices],rotation=40)
plt.show()
plt.show()



# Performance Tuning using Grid score
param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, param_grid,cv=5)
clf.fit(pcatrain,df_trainlabel)
print(clf.best_params_)


C_range = 10. ** np.arange(-3, 8)
gamma_range = 10. ** np.arange(-5, 4)

param_grid = dict(gamma=gamma_range, C=C_range)

grid = grid_search.GridSearchCV(SVC(), param_grid=param_grid)

grid.fit(pcatrain, df_trainlabel)

print("The best classifier is: ", grid.best_estimator_)

# Plot showing the relationship between C and Gamma
score_dict = grid.grid_scores_

# We extract just the scores
scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))

# Make a nice figure
cmap = cm.get_cmap("Spectral")
colors = cmap(a / b)
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=cmap)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.title("Parametre Comparison for C vs Gamma for kernel = rbf")
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.show()

# Accuracy has improved

svr = svm.SVC(kernel="rbf",C=1000,gamma=0.001)
cross_val_score(svr,pcatrain,df_trainlabel, cv=5).mean()
# Test Accuracy
scv = svr.fit(pcatrain, df_trainlabel)
y_pred = scv.predict(pcatest)
accuracy_score(y_pred, df_testlabel)
ac_score=accuracy_score(y_pred, df_testlabel)

cf_mat = confusion_matrix(df_testlabel, y_pred)
print("Accuracy: %f" %ac_score)
activities = le.classes_
activities
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(cf_mat, classes=activities,title="Confusion Matrix for Test data")




svm_final = SVC(kernel ="rbf",C=10,gamma=.1)
fstart_time = time.time()
svm_final.fit(df_traindata,df_trainlabel)
print("--- %s seconds ---" % (time.time() - fstart_time))


svm_final = SVC(kernel ="rbf",C=10,gamma=.1)
fstart_time = time.time()
svm_final.fit(pcatrain,df_trainlabel)
print("--- %s seconds ---" % (time.time() - fstart_time))
svm_pred = svm_final.predict_proba(pcatest)
skplt.metrics.plot_roc(pcatest, svm_pred)

skplt.metrics.plot_cumulative_gain(pcatest, svm_pred)

skplt.metrics.plot_lift_curve(pcatest, svm_pred)






# Time

# Naive 32.63s vs Optimized 0.25s


# =============================================================================
# def correlation(dataset, threshold):
#     col_corr = set() # Set of all the names of deleted columns
#     corr_matrix = dataset.corr().abs()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if corr_matrix.iloc[i, j] >= threshold:
#                 colname = corr_matrix.columns[i] # getting the name of column
#                 col_corr.add(colname)
#                 if colname in dataset.columns:
#                     del dataset[colname] # deleting the column from the dataset
#
#     return(dataset)
#
# cortrain=correlation(vartrain,0.95)
# cortrain.columns
# cortest[:,col for col in cortrain.columns]
# cortest=correlation(vartest,0.95)
# cortrain.shape
# cross_val_score(svc,cortrain,df_trainlabel, cv=5).mean()
# scv = svc.fit(cortrain, df_trainlabel)
# y_pred = scv.predict(cortest)
# accuracy_score(y_pred, df_testlabel)
#
# =============================================================================


