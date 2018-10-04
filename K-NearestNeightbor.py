import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sb
import six
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.metrics import precision_score, recall_score,f1_score,classification_report, confusion_matrix
from sklearn.base import clone
from time import time
plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')

def display_scores(scores):

	print("Scores:", scores)
	print("Mean:", scores.mean())
	print("Standard deviation:", scores.std())


pfile=open('featureherfinal','r')
data=pfile.read() 
pfile.close()
data=np.genfromtxt(six.StringIO(data)) #Se sobre entiende que los #delimitadores son espacios
data_pd=pd.DataFrame(data=data[0:,0:],index=data[0:,0],columns=data[0,0:])
data_pd_target=data_pd.drop([0.0], axis=1, inplace=False)
data_pd_label=data_pd[0.0].copy()

pfile=open('featurehervalidacionfinal','r')
dataval=pfile.read() 
pfile.close()
dataval=np.genfromtxt(six.StringIO(dataval)) #Se sobre entiende que los #delimitadores son espacios
data_pdval=pd.DataFrame(data=dataval[0:,0:],index=dataval[0:,0],columns=dataval[0,0:])
data_pd_targetval=data_pdval.drop([0.0], axis=1, inplace=False)
data_pd_labelval=data_pdval[0.0].copy()



file = open('EstadisticasKNN.txt','w')

#Creamos una figura
PCA2D=plt.figure()
pca2D = PCA(n_components = 2)
X2D = pca2D.fit_transform(data_pd_target)

pca2Dval=PCA(n_components = 2)
X2Dval = pca2Dval.fit_transform(data_pd_targetval)


colores=['blue','red','green','orange','yellow']
leyenda=['Alicate','Destornillador','Llave','Martillo','Taladro']
asignar_color=[]
for row in data_pd[0.0]:
    asignar_color.append(colores[int(row)])
plt.scatter(X2D[:,0], X2D[:,1], c=asignar_color,s=60)
for i in range(len(leyenda)):
	plt.scatter([],[],c=colores[i],label=leyenda[i])

# Mostramos en pantalla
plt.title('PCA 2D')
plt.legend()

# Create color maps
cmap_light = ListedColormap(['blue', 'red', 'green','orange','yellow'])
cmap_bold = ListedColormap(['blue', 'red', 'green','orange','yellow'])

h = .02  # step size in the mesh
metrics=[]
vecindad=[]
print ('PCA2D')
file.write("PCA2D\n")
accuracy=[]
modelo=1
for weights in ['uniform', 'distance']:
    for n_neighbors in range(4,9):
	    mean_kfold=[]
	    metrics.append(weights)
	    vecindad.append(n_neighbors)
	    print ('Modelo {!s}'.format(modelo))
	    file.write('Modelo {!s}\n'.format(modelo))
	    modelo = modelo+1
	    kfold=1
	    print (weights)
	    file.write('weights: {!s}\n'.format(weights))
	    # we create an instance of Neighbours Classifier and fit the data.
	    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	    #clf.fit(X2D, data_pd_label.values)
	    skfolds = StratifiedKFold(n_splits=7, random_state=42)
	    for train_index, test_index in skfolds.split(X2D, data_pd_label.values):
		contador=0
		clone_clf = clone(clf)
		X_train_folds = X2D[train_index]
		y_train_folds = (data_pd_label.values[train_index])
		X_test_fold = X2Dval
		y_test_fold = (data_pd_labelval.values)
		clone_clf.fit(X_train_folds, y_train_folds)
		print ('KFOLD: {!s}'.format(kfold))
		file.write('KFOLD: {!s}\n'.format(kfold))
		print("Accuracy score (training): {0:.3f}".format(clone_clf.score(X_train_folds,y_train_folds)))
		file.write("Accuracy score (training): {0:.3f}\n".format(clone_clf.score(X_train_folds,y_train_folds)))
		print("Accuracy score (validation): {0:.3f}".format(clone_clf.score(X_test_fold,y_test_fold)))
		file.write("Accuracy score (validation): {0:.3f}\n".format(clone_clf.score(X_test_fold,y_test_fold)))
		mean_kfold.append(clone_clf.score(X_test_fold,y_test_fold))
		predictions = clone_clf.predict(X_test_fold)
		print("Confusion Matrix:")
		file.write("Confusion Matrix:\n")
		print(confusion_matrix(y_test_fold, predictions))
		file.write('{!s}\n'.format(confusion_matrix(y_test_fold, predictions)))
		print("Classification Report")
		file.write("Classification Report\n")
		print(classification_report(y_test_fold, predictions))
		file.write('{!s}\n'.format(classification_report(y_test_fold, predictions)))
		kfold += 1
	    media=np.mean(np.array(mean_kfold))
	    print ('media: {!s}'.format(media))
	    file.write('media: {!s}\n'.format(media))
	    accuracy.append(media)

print ('#######################Analisis PCA2D#######################')
file.write('#######################Analisis PCA2D#######################\n')
best_metric=np.argmax(accuracy)
print ("Mejor modelo: {!s}".format(best_metric+1)+" {!s}".format(metrics[best_metric])+" {!s}".format(vecindad[best_metric]))
file.write("Mejor modelo: {!s}".format(best_metric+1)+" {!s}".format(metrics[best_metric])+" {!s}\n".format(vecindad[best_metric]))
KNN=neighbors.KNeighborsClassifier(n_neighbors, weights=metrics[best_metric])
KNN.fit(X2D,data_pd_label.values)
predictions = KNN.predict(X2Dval)
print("Accuracy score (training): {0:.3f}".format(KNN.score(X2D,data_pd_label)))
file.write("Accuracy score (training): {0:.3f}\n".format(KNN.score(X2D,data_pd_label)))
print("Accuracy score (validation): {0:.3f}".format(KNN.score(X2Dval,data_pd_labelval)))
file.write("Accuracy score (validation): {0:.3f}\n".format(KNN.score(X2Dval,data_pd_labelval)))
print("Confusion Matrix:")
file.write("Confusion Matrix:\n")
print(confusion_matrix(data_pd_labelval.values, predictions))
file.write('{!s}\n'.format(confusion_matrix(data_pd_labelval.values, predictions)))
print("Classification Report")
file.write("Classification Report\n")
print(classification_report(data_pd_labelval.values, predictions))
file.write('{!s}\n'.format(classification_report(data_pd_labelval.values, predictions)))


#Creamos una figura
pca3D = PCA(n_components = 3)
X3D = pca3D.fit_transform(data_pd_target)

pca3Dval = PCA(n_components = 3)
X3Dval = pca3Dval.fit_transform(data_pd_targetval)


fig_PCA = plt.figure()
ax=fig_PCA.gca(projection='3d')
sc=ax.scatter(X3D[:,0], X3D[:,1], X3D[:,2], c=asignar_color,s=60)
labelTups = [('Alicate', 0), ('Destornillador', 1), ('Llave', 2),('Martillo',3),('Taladro',4)]
custom_lines = [plt.Line2D([],[], ls="", marker='.',mec='k', mfc=c, mew=.1, ms=10) for c in colores]
ax.legend(custom_lines, [lt[0] for lt in labelTups],loc=1, bbox_to_anchor=(1.0, .5))

# Mostramos en pantalla
plt.title('PCA 3D')

metrics=[]
vecindad=[]
print ('PCA3D')
file.write("PCA3D\n")
accuracy=[]
modelo=1
for weights in ['uniform', 'distance']:
    for n_neighbors in range(4,9):
	    mean_kfold=[]
	    metrics.append(weights)
	    vecindad.append(n_neighbors)
	    print ('Modelo {!s}'.format(modelo))
	    file.write('Modelo {!s}\n'.format(modelo))
	    modelo = modelo+1
	    kfold=1
	    print (weights)
	    file.write('weights: {!s}\n'.format(weights))
	    # we create an instance of Neighbours Classifier and fit the data.
	    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	    #clf.fit(X2D, data_pd_label.values)
	    skfolds = StratifiedKFold(n_splits=7, random_state=42)
	    for train_index, test_index in skfolds.split(X3D, data_pd_label.values):
		contador=0
		clone_clf = clone(clf)
		X_train_folds = X3D[train_index]
		y_train_folds = (data_pd_label.values[train_index])
		X_test_fold = X3Dval
		y_test_fold = (data_pd_labelval.values)
		clone_clf.fit(X_train_folds, y_train_folds)
		print ('KFOLD: {!s}'.format(kfold))
		file.write('KFOLD: {!s}\n'.format(kfold))
		print("Accuracy score (training): {0:.3f}".format(clone_clf.score(X_train_folds,y_train_folds)))
		file.write("Accuracy score (training): {0:.3f}\n".format(clone_clf.score(X_train_folds,y_train_folds)))
		print("Accuracy score (validation): {0:.3f}".format(clone_clf.score(X_test_fold,y_test_fold)))
		file.write("Accuracy score (validation): {0:.3f}\n".format(clone_clf.score(X_test_fold,y_test_fold)))
		mean_kfold.append(clone_clf.score(X_test_fold,y_test_fold))
		predictions = clone_clf.predict(X_test_fold)
		print("Confusion Matrix:")
		file.write("Confusion Matrix:\n")
		print(confusion_matrix(y_test_fold, predictions))
		file.write('{!s}\n'.format(confusion_matrix(y_test_fold, predictions)))
		print("Classification Report")
		file.write("Classification Report\n")
		print(classification_report(y_test_fold, predictions))
		file.write('{!s}\n'.format(classification_report(y_test_fold, predictions)))
		kfold += 1
	    media=np.mean(np.array(mean_kfold))
	    print 'media: {!s}'.format(media)
	    file.write('media: {!s}\n'.format(media))
	    accuracy.append(media)

print ('#######################Analisis PCA3D#######################')
file.write('#######################Analisis PCA3D#######################\n')
best_metric=np.argmax(accuracy)
print ("Mejor modelo: {!s}".format(best_metric+1)+" {!s}".format(metrics[best_metric])+" {!s}".format(vecindad[best_metric]))
file.write("Mejor modelo: {!s}".format(best_metric+1)+" {!s}".format(metrics[best_metric])+" {!s}\n".format(vecindad[best_metric]))
KNN=neighbors.KNeighborsClassifier(n_neighbors, weights=metrics[best_metric])
KNN.fit(X3D,data_pd_label.values)
predictions = KNN.predict(X3Dval)
print("Accuracy score (training): {0:.3f}".format(KNN.score(X3D,data_pd_label)))
file.write("Accuracy score (training): {0:.3f}\n".format(KNN.score(X3D,data_pd_label)))
print("Accuracy score (validation): {0:.3f}".format(KNN.score(X3Dval,data_pd_labelval)))
file.write("Accuracy score (validation): {0:.3f}\n".format(KNN.score(X3Dval,data_pd_labelval)))
print("Confusion Matrix:")
file.write("Confusion Matrix:\n")
print(confusion_matrix(data_pd_labelval.values, predictions))
file.write('{!s}\n'.format(confusion_matrix(data_pd_labelval.values, predictions)))
print("Classification Report")
file.write("Classification Report\n")
print(classification_report(data_pd_labelval.values, predictions))
file.write('{!s}\n'.format(classification_report(data_pd_labelval.values, predictions)))

metrics=[]
vecindad=[]
print ('PCA334D')
file.write("PCA334D\n")
Estadisticos_Accuracy=[]
Estadisticos_F1_Score=[]
Estadisticos_Accuracy_Std=[]
Estadisticos_F1_Score_Std=[]
Estadisticos_Time_Mean=[]
Estadisticos_Time_Std=[]
modelo=1
for weights in ['uniform', 'distance']:
    for n_neighbors in range(4,9):
	    Accuracy_kfold=[]
	    F1_Score_kfold=[]
	    Time_kfold=[]
	    metrics.append(weights)
	    vecindad.append(n_neighbors)
	    print ('Modelo {!s}'.format(modelo))
	    file.write('Modelo {!s}\n'.format(modelo))
	    modelo = modelo+1
	    kfold=1
	    print (weights)
	    file.write('weights: {!s}\n'.format(weights))
	    # we create an instance of Neighbours Classifier and fit the data.
	    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	    #clf.fit(X2D, data_pd_label.values)
	    skfolds = StratifiedKFold(n_splits=7, random_state=42)
	    for train_index, test_index in skfolds.split(data_pd_target.values, data_pd_label.values):
		contador = 0
		clone_clf = clone(clf)
		X_train_folds = data_pd_target.values[train_index]
		y_train_folds = (data_pd_label.values[train_index])
		X_test_fold = data_pd_targetval.values
		y_test_fold = (data_pd_labelval.values)
		t1=time()
		clone_clf.fit(X_train_folds, y_train_folds)
		t2=time()
		print ('KFOLD: {!s}'.format(kfold))
		file.write('KFOLD: {!s}\n'.format(kfold))
		print("Accuracy score (training): {0:.3f}".format(clone_clf.score(X_train_folds,y_train_folds)))
		file.write("Accuracy score (training): {0:.3f}\n".format(clone_clf.score(X_train_folds,y_train_folds)))
		print("Accuracy score (validation): {0:.3f}".format(clone_clf.score(X_test_fold,y_test_fold)))
		file.write("Accuracy score (validation): {0:.3f}\n".format(clone_clf.score(X_test_fold,y_test_fold)))
		Accuracy_kfold.append(clone_clf.score(X_test_fold,y_test_fold))
		predictions = clone_clf.predict(X_test_fold)
		print("Confusion Matrix:")
		file.write("Confusion Matrix:\n")
		print(confusion_matrix(y_test_fold, predictions))
		file.write('{!s}\n'.format(confusion_matrix(y_test_fold, predictions)))
		print("Classification Report")
		file.write("Classification Report\n")
		print(classification_report(y_test_fold, predictions))
		file.write('{!s}\n'.format(classification_report(y_test_fold, predictions)))
		F1S=f1_score(np.argmax(self.data_pd_targetval.values, axis=1),predictions,average=None)
		F1_Score_kfold.append(((F1S[0]*119+F1S[1]*140+F1S[2]*96+F1S[3]*109+F1S[4]*111)/575))
		Time_kfold.append((t2-t1)/60)
		kfold += 1
	    Accuracy_Mean=np.mean(np.array(Accuracy_kfold))
	    Accuracy_Std=np.std(np.array(Accuracy_kfold))
	    F1_Score_Mean=np.mean(np.array(F1_Score_kfold))
	    F1_Score_Std=np.std(np.array(F1_Score_kfold))
	    Time_Mean=np.mean(np.array(Time_kfold))
	    Time_Std=np.std(np.array(Time_kfold))
	    print ('Accuracy_Mean: {!s}'.format(Accuracy_Mean))
	    file.write('Accuracy_Mean: {!s}\n'.format(Accuracy_Mean))
	    print ('Accuracy_Std: {!s}'.format(Accuracy_Std))
	    file.write('Accuracy_Std: {!s}\n'.format(Accuracy_Std))
	    print ('F1_Score_Mean: {!s}'.format(F1_Score_Mean))
	    file.write('F1_Score_Mean: {!s}\n'.format(F1_Score_Mean))
	    print ('F1_Score_Std: {!s}'.format(F1_Score_Std))
	    file.write('F1_Score_Std: {!s}\n'.format(F1_Score_Std))
	    print ('Time_Mean: {!s}'.format(Time_Mean))
            file.write('Time_Mean: {!s}\n'.format(Time_Mean))
	    print ('Time_Std: {!s}'.format(Time_Std))
            file.write('Time_Std: {!s}\n'.format(Time_Std))
	    Estadisticos_Accuracy_Mean.append(Accuracy_Mean)
	    Estadisticos_F1_Score_Mean.append(F1_Score_Mean)
	    Estadisticos_Accuracy_Std.append(Accuracy_Std)
	    Estadisticos_F1_Score_Std.append(F1_Score_Std)
	    Estadisticos_Time_Mean.append(Time_Mean)
	    Estadisticos_Time_Std.append(Time_Std)

print ('#######################Analisis PCA334D#######################')
file.write('#######################Analisis PCA334D#######################\n')
best_metric=np.argmax(Estadisticos_F1_Score_Mean)
print ("Mejor modelo: {!s}".format(best_metric+1)+" Weight: {!s}".format(metrics[best_metric])+" Nearest: {!s}".format(vecindad[best_metric]))
file.write("Mejor modelo: {!s}".format(best_metric+1)+" Weight: {!s}".format(metrics[best_metric])+" Nearest: {!s}\n".format(vecindad[best_metric]))
print ("F1_Score_Mean: {!s}".format(Estadisticos_F1_Score_Mean[best_metric])+" F1_Score_Std: {!s}".format(Estadisticos_F1_Score_Std[best_metric]))
file.write("F1_Score_Mean: {!s}".format(Estadisticos_F1_Score_Mean[best_metric])+" F1_Score_Std: {!s}\n".format(Estadisticos_F1_Score_Std[best_metric]))
print ("Time_Mean: {!s}".format(Estadisticos_Time_Mean[best_metric])+" Time_Std: {!s}".format(Estadisticos_Time_Std[best_metric]))
file.write("Time_Mean: {!s}".format(Estadisticos_Time_Mean[best_metric])+" Time_Std: {!s}\n".format(Estadisticos_Time_Std[best_metric]))
KNN=neighbors.KNeighborsClassifier(n_neighbors, weights=metrics[best_metric])
KNN.fit(data_pd_target.values,data_pd_label.values)
predictions = KNN.predict(data_pd_targetval.values)
print("Accuracy score (training): {0:.3f}".format(KNN.score(data_pd_target.values,data_pd_label)))
file.write("Accuracy score (training): {0:.3f}\n".format(KNN.score(data_pd_target.values,data_pd_label)))
print("Accuracy score (validation): {0:.3f}".format(KNN.score(data_pd_targetval.values,data_pd_labelval)))
file.write("Accuracy score (validation): {0:.3f}\n".format(KNN.score(data_pd_targetval.values,data_pd_labelval)))
print("Confusion Matrix:")
file.write("Confusion Matrix:\n")
print(confusion_matrix(data_pd_labelval.values, predictions))
file.write('{!s}\n'.format(confusion_matrix(data_pd_labelval.values, predictions)))
print("Classification Report")
file.write("Classification Report\n")
print(classification_report(data_pd_labelval.values, predictions)) 
file.write('{!s}\n'.format(classification_report(data_pd_labelval.values, predictions)))   
file.close()


plt.show()
