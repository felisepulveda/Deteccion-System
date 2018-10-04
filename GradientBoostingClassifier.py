import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import six
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 
from time import time


pfile=open('featureheraux.txt','r')
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

file = open('EstadisticasGBC.txt','w')
Estadisticos_Accuracy_Mean=[]
Estadisticos_F1_Score_Mean=[]
Estadisticos_Accuracy_Std=[]
Estadisticos_F1_Score_Std=[]
Estadisticos_Time_Mean=[]
Estadisticos_Time_Std=[]
modelo=1
a=[]
b=[]
c=[]
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
depth=[3,4,5]
for learning_rate in learning_rates:
    for depths in depth:
	    for estimators in range(1,60):
		Accuracy_kfold=[]
		F1_Score_kfold=[]
		Time_kfold=[]
		print ("############################################################")
		file.write('##################################################################################\n')
		print ('Modelo {!s}'.format(modelo))
		file.write('Modelo {!s}\n'.format(modelo))
		modelo = modelo+1
		print ('learning_rate: {!s}'.format(learning_rate)+' estimators: {!s}'.format(estimators)+' depth: {!s}'.format(depths))
		file.write('learning_rate: {!s}'.format(learning_rate)+' estimators: {!s}'.format(estimators)+' depth: {!s}\n'.format(depths))
		kfold=1
		clf = GradientBoostingClassifier(n_estimators=estimators, learning_rate = learning_rate, max_depth = depths, random_state = 0)
		skfolds = StratifiedKFold(n_splits=7, random_state=42)
		for train_index, test_index in skfolds.split(data_pd_target.values, data_pd_label.values):
		    clone_clf = clone(clf)
		    t1=time()
		    clone_clf.fit(data_pd_target.values[train_index],data_pd_label.values[train_index])
		    t2=time()
		    print ('KFOLD: {!s}'.format(kfold))
		    file.write('KFOLD: {!s}\n'.format(kfold))
		    print("Accuracy score (training): {0:.3f}".format(clone_clf.score(data_pd_target.values[train_index],data_pd_label.values[train_index])))
		    file.write("Accuracy score (training): {0:.3f}\n".format(clone_clf.score(data_pd_target.values[train_index],data_pd_label.values[train_index])))
		    print("Accuracy score (validation): {0:.3f}".format(clone_clf.score(data_pd_targetval.values,data_pd_labelval.values)))
		    file.write("Accuracy score (validation): {0:.3f}\n".format(clone_clf.score(data_pd_targetval.values,data_pd_labelval.values)))
		    Accuracy_kfold.append(clone_clf.score(data_pd_targetval.values,data_pd_labelval.values))
		    predictions = clone_clf.predict(data_pd_targetval.values)
		    print("Confusion Matrix:")
		    file.write("Confusion Matrix:\n")
		    print(confusion_matrix(data_pd_labelval.values, predictions))
		    file.write('{!s}\n'.format(confusion_matrix(data_pd_labelval.values, predictions)))
		    print("Classification Report:")
		    file.write("Classification Report:\n")
		    print(classification_report(data_pd_labelval.values, predictions))
		    file.write('{!s}\n'.format(classification_report(data_pd_labelval.values, predictions)))
		    F1S=f1_score(np.argmax(self.data_pd_targetval.values, axis=1),predictions,average=None)
		    F1_Score_kfold.append(((F1S[0]*119+F1S[1]*140+F1S[2]*96+F1S[3]*109+F1S[4]*111)/575))
		    Time_kfold.append((t2-t1)/60)
		    kfold += 1
		a.append(learning_rate)
		b.append(estimators)
		c.append(depths)
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
		print ('Time_Std: {!s}'.format(Time_Std))
                file.write('Time_Std: {!s}\n'.format(Time_Std))
		Estadisticos_Accuracy_Mean.append(Accuracy_Mean)
		Estadisticos_F1_Score_Mean.append(F1_Score_Mean)
		Estadisticos_Accuracy_Std.append(Accuracy_Std)
		Estadisticos_F1_Score_Std.append(F1_Score_Std)
		Estadisticos_Time_Mean.append(Time_Mean)
	        Estadisticos_Time_Std.append(Time_Std)
		print ("############################################################")
		file.write('##################################################################################\n')

"""best_learning_rate=np.argmax(accuracy)
GBC=GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rates[best_learning_rate], max_depth = 2, random_state = 0)
GBC.fit(data_pd_target.values,data_pd_label.values)
errors = [np.mean(np.equal(data_pd_labelval.values, y_pred)) for y_pred in GBC.staged_predict(data_pd_targetval.values)]
best_n_estimators = np.argmax(errors)"""

best=np.argmax(Estadisticos_F1_Score_Mean)
print ("Mejor modelo: {!s}".format(best+1))
file.write("Mejor modelo: {!s}\n".format(best+1))
print ("learning rate: {!s}".format(a[best])+" n_estimators: {!s}".format(b[best])+" depth: {!s}".format(c[best]))
file.write("learning rate: {!s}".format(a[best])+" n_estimators: {!s}".format(b[best])+" depth: {!s}\n".format(c[best]))
print ("F1_Score_Mean: {!s}".format(Estadisticos_F1_Score_Mean[best])+" F1_Score_Std: {!s}".format(Estadisticos_F1_Score_Std[best]))
file.write("F1_Score_Mean: {!s}".format(Estadisticos_F1_Score_Mean[best])+" F1_Score_Std: {!s}\n".format(Estadisticos_F1_Score_Std[best]))
print ("Time_Mean: {!s}".format(Estadisticos_Time_Mean[best])+" Time_Std: {!s}".format(Estadisticos_Time_Std[best]))
file.write("Time_Mean: {!s}".format(Estadisticos_Time_Mean[best])+" Time_Std: {!s}\n".format(Estadisticos_Time_Std[best]))
GBC_best = GradientBoostingClassifier(n_estimators=b[best], learning_rate = a[best], max_depth = c[best], random_state = 0)
GBC_best.fit(data_pd_target.values,data_pd_label.values)
predictions = GBC_best.predict(data_pd_targetval.values)
print("Accuracy score (training): {0:.3f}".format(GBC_best.score(data_pd_target.values,data_pd_label.values)))
file.write("Accuracy score (training): {0:.3f}\n".format(GBC_best.score(data_pd_target.values,data_pd_label.values)))
print("Accuracy score (validation): {0:.3f}".format(GBC_best.score(data_pd_targetval.values,data_pd_labelval.values)))
file.write("Accuracy score (validation): {0:.3f}\n".format(GBC_best.score(data_pd_targetval.values,data_pd_labelval.values)))
print("Confusion Matrix:")
file.write("Confusion Matrix:\n")
print(confusion_matrix(data_pd_labelval.values, predictions))
file.write('{!s}\n'.format(confusion_matrix(data_pd_labelval.values, predictions)))
print("Classification Report:")
file.write("Classification Report:\n")
print(classification_report(data_pd_labelval.values, predictions))
file.write('{!s}\n'.format(classification_report(data_pd_labelval.values, predictions)))
file.close()
