#!/usr/bin/env python
# -*- coding: cp1252 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
import six
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,StratifiedKFold 
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report, confusion_matrix
from datetime import datetime
from time import time

class DNN:

    def __init__(self,archivo):

        self.num_classes = 5
        self.x_width = 334
        self.col1cost = 1
        self.col2cost = 1
        self.col3cost = 1
        self.col4cost = 1
        self.col5cost = 1
        self.archivo=archivo

    def Lee_Datos(self): 

        pfile=open('featureherfinal','r')
        data=pfile.read() 
        pfile.close()
        data=np.genfromtxt(six.StringIO(data)) #Se sobre entiende que los #delimitadores son espacios
        data_pd=pd.DataFrame(data=data[0:,0:],index=data[0:,0],columns=data[0,0:])
        self.data_pd_target_multiclass=data_pd[0.0].copy()
        data_pd[1.0]=np.floor(data_pd[0.0] / 5)
        data_pd[2.0]=np.floor(data_pd[0.0] / 5)
        data_pd[3.0]=np.floor(data_pd[0.0] / 5)
        data_pd[4.0]=np.floor(data_pd[0.0] / 5)
        self.data_pd_target = data_pd[[0.0,1.0,2.0,3.0,4.0]].copy()
        self.data_pd_target[1.0].where(self.data_pd_target[0.0] != 1.0, 1.0, inplace=True)
        self.data_pd_target[2.0].where(self.data_pd_target[0.0] != 2.0, 1.0, inplace=True)
        self.data_pd_target[3.0].where(self.data_pd_target[0.0] != 3.0, 1.0, inplace=True)
        self.data_pd_target[4.0].where(self.data_pd_target[0.0] != 4.0, 1.0, inplace=True)
        self.data_pd_target[0.0].where(self.data_pd_target[0.0] == 0.0, 1.0, inplace=True)
        self.data_pd_target[0.0] = np.ceil(self.data_pd_target[0.0] -1.0)
        self.data_pd_target[0.0].where(self.data_pd_target[0.0] == 0.0, 1.0, inplace=True)
        self.data_pd_data = data_pd.drop([0.0,1.0,2.0,3.0,4.0], axis=1)

        pfile=open('featurehervalidacionfinal','r')
        dataval=pfile.read() 
        pfile.close()
        dataval=np.genfromtxt(six.StringIO(dataval)) #Se sobre entiende que los #delimitadores son espacios
        data_pdval=pd.DataFrame(data=dataval[0:,0:],index=dataval[0:,0],columns=dataval[0,0:])
        data_pdval[1.0]=np.floor(data_pdval[0.0] / 5)
        data_pdval[2.0]=np.floor(data_pdval[0.0] / 5)
        data_pdval[3.0]=np.floor(data_pdval[0.0] / 5)
        data_pdval[4.0]=np.floor(data_pdval[0.0] / 5)
        self.data_pd_targetval = data_pdval[[0.0,1.0,2.0,3.0,4.0]].copy()
        self.data_pd_targetval[1.0].where(self.data_pd_targetval[0.0] != 1.0, 1.0, inplace=True)
        self.data_pd_targetval[2.0].where(self.data_pd_targetval[0.0] != 2.0, 1.0, inplace=True)
        self.data_pd_targetval[3.0].where(self.data_pd_targetval[0.0] != 3.0, 1.0, inplace=True)
        self.data_pd_targetval[4.0].where(self.data_pd_targetval[0.0] != 4.0, 1.0, inplace=True)
        self.data_pd_targetval[0.0].where(self.data_pd_targetval[0.0] == 0.0, 1.0, inplace=True)
        self.data_pd_targetval[0.0] = np.ceil(self.data_pd_targetval[0.0] -1.0)
        self.data_pd_targetval[0.0].where(self.data_pd_targetval[0.0] == 0.0, 1.0, inplace=True)
        self.data_pd_dataval = data_pdval.drop([0.0,1.0,2.0,3.0,4.0], axis=1)


    def fetch_batch_all(self,epoch, batch_index, batch_size):
        m, n = self.data_pd_data.shape
        data_pd_data_aux=self.data_pd_data.values
        data_pd_target_aux=self.data_pd_target.values
        #housing_data_plus_bias_target = np.c_[np.ones((m, 1)), housing.target]
        #housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
        if (batch_index == int(self.data_pd_data.values.shape[0]/self.batch_size)-1):
            data_pd_data_aux = data_pd_data_aux[batch_index*batch_size:,0:]
            data_pd_target_aux = data_pd_target_aux[batch_index*batch_size:,]
        else:    
            data_pd_data_aux = data_pd_data_aux[batch_index*batch_size:(batch_index+1)*batch_size,0:]
            data_pd_target_aux = data_pd_target_aux[batch_index*batch_size:(batch_index+1)*batch_size,]
        #housing_data_plus_bias=pd.DataFrame(data=housing_data_plus_bias[0:,0:],index=housing_data_plus_bias[0:,0],columns=housing_data_plus_bias[0,0:]) 
        #scaled_housing_data_plus_bias = num_pipeline.fit_transform(housing_data_plus_bias)
        #X_batch = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
        #y_batch = tf.constant(housing_data_plus_bias_target.reshape(-1, 1), dtype=tf.float32, name="y")
        X_batch = data_pd_data_aux
        y_batch = data_pd_target_aux
        return X_batch, y_batch

    def fetch_batch(self,epoch, batch_index, batch_size):
        m, n = self.data_pd_data.shape
        data_pd_data_aux=self.data_pd_data.values[self.train_index]
        data_pd_target_aux=self.data_pd_target.values[self.train_index]
        #housing_data_plus_bias_target = np.c_[np.ones((m, 1)), housing.target]
        #housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
        if (batch_index == int(self.data_pd_data.values[self.train_index].shape[0]/self.batch_size)-1):
            data_pd_data_aux = data_pd_data_aux[batch_index*batch_size:,0:]
            data_pd_target_aux = data_pd_target_aux[batch_index*batch_size:,]
        else:    
            data_pd_data_aux = data_pd_data_aux[batch_index*batch_size:(batch_index+1)*batch_size,0:]
            data_pd_target_aux = data_pd_target_aux[batch_index*batch_size:(batch_index+1)*batch_size,]
        #housing_data_plus_bias=pd.DataFrame(data=housing_data_plus_bias[0:,0:],index=housing_data_plus_bias[0:,0],columns=housing_data_plus_bias[0,0:]) 
        #scaled_housing_data_plus_bias = num_pipeline.fit_transform(housing_data_plus_bias)
        #X_batch = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
        #y_batch = tf.constant(housing_data_plus_bias_target.reshape(-1, 1), dtype=tf.float32, name="y")
        X_batch = data_pd_data_aux
        y_batch = data_pd_target_aux
        return X_batch, y_batch

    # Tensorflow convinience functions
    def weight_variable(self,shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self,shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean_'+var.op.name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev_'+var.op.name, stddev)
            tf.summary.scalar('max_'+var.op.name, tf.reduce_max(var))
            tf.summary.scalar('min_'+var.op.name, tf.reduce_min(var))
            tf.summary.histogram('histogram_'+var.op.name, var)


    def multilayer_perceptron(self):

        w_f=[]
        b_f=[]
        h_fc=[]
        h_fc_drop=[]

        # Fully connected layer 1:
        with tf.name_scope('input_layer') as scope:
            w_fc1 = self.weight_variable([self.x_width, self.num_neurons], 'weight_input')   # weights
            self.variable_summaries(w_fc1)
            b_fc1 = self.bias_variable([self.num_neurons], 'bias_input')  # biases
            self.variable_summaries(b_fc1)
            h_fc1 = tf.nn.relu(tf.matmul(self.x, w_fc1) + b_fc1, name='af_relu_input') # activation
            tf.summary.histogram('activations_input', h_fc1)
            h_fc_drop.append(tf.nn.dropout(h_fc1, self.keep_prob, name='dropout_input')) # dropout

        for i in range(self.layers):
            with tf.name_scope('hidden_{!s}'.format(i+1)) as scope:
                w_f.append(self.weight_variable([self.num_neurons, self.num_neurons], 'weight_h{!s}'.format(i+1)))
                self.variable_summaries(w_f[i])
                b_f.append(self.bias_variable([self.num_neurons], 'bias_h{!s}'.format(i+1)))
                self.variable_summaries(b_f[i])
                h_fc.append(tf.nn.relu(tf.matmul(h_fc_drop[i], w_f[i]) + b_f[i], name='af_relu_h{!s}'.format(i+1)))
                tf.summary.histogram('activations_hidden{!s}'.format(i+1), h_fc[i])
                h_fc_drop.append(tf.nn.dropout(h_fc[i], self.keep_prob, name='dropout_h{!s}'.format(i+1)))

        # Readout layer
        with tf.name_scope('read_out') as scope:
            w_fc_out = self.weight_variable([self.num_neurons, self.num_classes], 'weight_out')
            self.variable_summaries(w_fc_out)
            b_fc_out = self.bias_variable([self.num_classes], 'bias_out')
            self.variable_summaries(b_fc_out)
            # The softmax function will make probabilties of Good vs Bad score at the output
            self.logits = tf.matmul(h_fc_drop[-1], w_fc_out) + b_fc_out
            y_ = tf.nn.softmax(self.logits, name='af_softmax')
            tf.summary.histogram('activations_out', y_)
        return y_

    def create_graph(self,epochs,num_neurons,learning_rate,batch_size,layers,drop_out,optimizador,train_index,test_index):
        self.g = tf.Graph()
        self.n_epochs = epochs #tuning
        self.num_neurons = num_neurons #tuning
        self.learning_rate = learning_rate #tuning
        self.batch_size = batch_size #tuning
        self.layers=layers #tuning
        self.drop_out = drop_out #tuning
        self.optimizador=optimizador #tuning
        self.now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root_logdir = "tf_logs"
        self.logdir = "{}/run-{}/".format(self.root_logdir, self.now)
        self.train_index=train_index
        self.test_index=test_index
        with self.g.as_default() as g:    

            ### Placeholders ###
            with tf.name_scope('placeholder') as scope:
                self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')
                self.x = tf.placeholder(tf.float32, [None, self.x_width], name='X') # Placeholder values
                self.keep_prob = tf.placeholder("float", name='keep_prob') # Placeholder values

            ### Create Network ###
            self.y_ = self.multilayer_perceptron()

            ### Accuracy Metrics trainning ###
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy_summary', self.accuracy)

                """with tf.name_scope('precision_score'):
                    self.precision=precision_score(tf.argmax(self.y,1),tf.argmax(self.y_,1),average=None)

                tf.summary.scalar('precision_summary', self.precision)
                with tf.name_scope('f1_score'):
                    self.f1=f1_score(tf.argmax(self.y, 1),tf.argmax(self.y_, 1),average=None)

                tf.summary.scalar('f1_summary', self.f1)
                with tf.name_scope('recall_score'):
                    self.recall=recall_score(tf.argmax(self.y, 1),tf.argmax(self.y_, 1),average=None)

                tf.summary.scalar('recall_summary', self.recall)"""
                self.correct_answer = tf.argmax(self.y_, 1)






            ### Customized Weighted Loss ###
            square_diff = tf.square(self.y - self.y_) # compute the prediction difference
            col1, col2, col3, col4, col5 = tf.split(square_diff,5,1) # split the (m,2) vector in two (m,1) vectors
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
            weights = tf.trainable_variables()

            with tf.name_scope("cost_function") as scope:
                regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
                costwise_loss = self.col1cost*tf.reduce_sum(col1) + self.col2cost*tf.reduce_sum(col2) +self.col3cost*tf.reduce_sum(col3) + self.col4cost*tf.reduce_sum(col4) +self.col5cost*tf.reduce_sum(col5)
                costwise_loss = costwise_loss + regularization_penalty
                tf.summary.scalar('cost_function_summary', costwise_loss)

            # Train the algorithm using gradient descent
            with tf.name_scope("train") as scope:
                if self.optimizador==1:
                    self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(costwise_loss)
                elif self.optimizador==2:
                    self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(costwise_loss)
            ### Write Summary Out ###
            self.merged = tf.summary.merge_all()

            self.init = tf.global_variables_initializer()  
            self.saver = tf.train.Saver()

            self.file_writer = tf.summary.FileWriter(self.logdir,g)

            #return g, train_step, x, y, keep_prob, y_, merged

    def Training(self,modelo,booleana):
        with self.g.as_default() as g:
            with tf.Session() as sess:
                self.init.run()
                #kfold=1
                #skfolds = StratifiedKFold(n_splits=7, random_state=42)
                #for train_index, test_index in skfolds.split(self.data_pd_data.values,self.data_pd_target_multiclass.values):
                for epoch in range(self.n_epochs):
                    if booleana==1:
                        for batch_index in range(int(self.data_pd_data.values.shape[0]/self.batch_size)):
                            X_batch, y_batch = self.fetch_batch_all(epoch, batch_index, self.batch_size)
                            summary,_=sess.run([self.merged,self.train_step],feed_dict={self.x: X_batch, self.y: y_batch,self.keep_prob: self.drop_out})
                    else:
                        for batch_index in range(int(self.data_pd_data.values[self.train_index].shape[0]/self.batch_size)):
                            X_batch, y_batch = self.fetch_batch(epoch, batch_index, self.batch_size)
                            summary,_=sess.run([self.merged,self.train_step],feed_dict={self.x: X_batch, self.y: y_batch,self.keep_prob: self.drop_out})
                    #self.file_writer.add_summary(summary, epoch)    
                    accuracytrain = self.accuracy.eval(feed_dict={self.x: self.data_pd_data.values, self.y: self.data_pd_target.values,self.keep_prob: 1.0})
                    #acc_test = self.accuracy.eval(feed_dict={self.x: self.data_pd_data.values,self.y: self.data_pd_target.values,self.keep_prob: 1.0})
                    accuracyval = self.accuracy.eval(feed_dict={self.x: self.data_pd_dataval.values,self.y: self.data_pd_targetval.values,self.keep_prob: 1.0})
                    """precision = self.precision.eval(feed_dict={self.x: self.data_pd_dataval.values,self.y: self.data_pd_targetval.values,self.keep_prob: 1.0})
                    recall = self.recall.eval(feed_dict={self.x: self.data_pd_dataval.values,self.y: self.data_pd_targetval.values,self.keep_prob: 1.0})
                    f1 = self.f1.eval(feed_dict={self.x: self.data_pd_dataval.values,self.y: self.data_pd_targetval.values,self.keep_prob: 1.0})"""
                    #print(epoch, "Test accuracy:", accuracy,"Test precision:", precision,"Test recall:", recall,"Test f1:", f1)
		    print(epoch, "Train accuracy:", accuracytrain)
		    self.archivo.write('{!s} '.format(epoch)+'Train accuracy: {!s}\n'.format(accuracytrain))
                    print(epoch, "Test accuracy:", accuracyval)
                    self.archivo.write('{!s} '.format(epoch)+'Test accuracy: {!s}\n'.format(accuracyval))
                    y_pred=self.correct_answer.eval(feed_dict={self.x: self.data_pd_dataval.values,self.y: self.data_pd_targetval.values,self.keep_prob: 1.0})
                    #print (y_pred)
                    print("Confusion Matrix:")
                    file.write("Confusion Matrix:\n")
                    print(confusion_matrix(np.argmax(self.data_pd_targetval.values, axis=1), y_pred))
                    file.write('{!s}\n'.format(confusion_matrix(np.argmax(self.data_pd_targetval.values, axis=1), y_pred)))
                    """print (precision_score(np.argmax(self.data_pd_targetval.values, axis=1),y_pred,average=None))
                    self.archivo.write('{!s}\n'.format(precision_score(np.argmax(self.data_pd_targetval.values, axis=1),y_pred,average=None)))
                    #print (np.argmax(self.data_pd_targetval.values, axis=1))
                    print (recall_score(np.argmax(self.data_pd_targetval.values, axis=1),y_pred,average=None))
                    self.archivo.write('{!s}\n'.format(recall_score(np.argmax(self.data_pd_targetval.values, axis=1),y_pred,average=None)))"""
		    """print("f1-score:")
		    file.write("f1-score:\n")
                    print (f1_score(np.argmax(self.data_pd_targetval.values, axis=1),y_pred,average=None))
                    self.archivo.write('{!s}\n'.format(f1_score(np.argmax(self.data_pd_targetval.values, axis=1),y_pred,average=None)))"""
                    print("Classification Report:")
                    file.write("Classification Report:\n")
                    print(classification_report(np.argmax(self.data_pd_targetval.values, axis=1), y_pred))
                    file.write('{!s}\n'.format(classification_report(np.argmax(self.data_pd_targetval.values, axis=1), y_pred)))
                #print ('##############################################################################')
                #print (kfold)     
                #kfold = (kfold + 1)  
                if booleana==1:
                    save_path = self.saver.save(sess, "./final{!s}.ckpt".format(modelo))
                self.file_writer.close()
                #self.Test(kfold)
		F1S=f1_score(np.argmax(self.data_pd_targetval.values, axis=1),y_pred,average=None)
                return accuracyval,((F1S[0]*119+F1S[1]*140+F1S[2]*96+F1S[3]*109+F1S[4]*111)/575)


    def Test(self,kfold): # Para probar con los datos del set de prueba
        with self.g.as_default() as g:   
            with tf.Session() as sess:
                self.saver.restore(sess, "./final{!s}.ckpt".format(kfold))
                Z = self.logits.eval(feed_dict={self.x: self.data_pd_dataval.values,self.keep_prob:1.0})
                y_pred = np.argmax(Z, axis=1)
                data_pd_test_target = np.argmax(data_pd_targetval.values, axis=1)
            print (kfold)
            print (confusion_matrix(data_pd_test_target, y_pred))
            print (precision_score(data_pd_test_target,y_pred,average=None))
            print (recall_score(data_pd_test_target,y_pred,average=None))
            print (f1_score(data_pd_test_target,y_pred,average=None))

    def Entrada(self,feature):
        feature = np.asarray(feature) #Lo transformo de list a numpy array
        feature=np.array(feature)[np.newaxis]
        Z = self.y_.eval(feed_dict={self.x: feature,self.keep_prob:1.0})
        y_pred = np.argmax(Z, axis=1)
        return y_pred

if __name__ == "__main__":
    file = open('EstadisticasFullyConected.txt','w')
    Estadisticos_Accuracy_Mean=[]
    Estadisticos_F1_Score_Mean=[]
    Estadisticos_Accuracy_Std=[]
    Estadisticos_F1_Score_Std=[]
    Estadisticos_Time_Mean=[]
    Estadisticos_Time_Std=[]
    BM=[]
    DNN=DNN(file)
    DNN.Lee_Datos()
    # Aca se ingresan los parametros para el grid
    epochs=[2000]
    num_neurons=[400,500,600]
    learning_rate=[0.001]
    batch_size=[200]
    layers=[9,8,7]
    drop_out=[0.6]
    optimizador=[2] # optimizador 1: gradient descent ; optimizador 2: adam
    # Fin ingreso parametros
    modelo=1
    for a,b,c,d,e,f,g in [(a,b,c,d,e,f,g) for a in epochs for b in num_neurons for c in learning_rate for d in batch_size for e in layers for f in drop_out for g in optimizador]:
        skfolds = StratifiedKFold(n_splits=7, random_state=42)
        kfold=1
        BM.append([a,b,c,d,e,f,g])
        Accuracy_kfold=[]
	F1_Score_kfold=[]
	Time_kfold=[]
        print ('##################################################################################')
        file.write('##################################################################################\n')
        print ('Numero del modelo',modelo)
        file.write('Numero del modelo {!s}\n'.format(modelo))
        for train_index, test_index in skfolds.split(DNN.data_pd_data.values,DNN.data_pd_target_multiclass.values):
            print ('Kfold: {!s}'.format(kfold))
            file.write('Kfold: {!s}\n'.format(kfold))
            print (a,b,c,d,e,f,g,kfold)
            file.write('{!s},{!s},{!s},{!s},{!s},{!s},{!s},{!s}\n'.format(a,b,c,d,e,f,g,kfold))
            print ('Iteraciones')
            file.write('Iteraciones\n')
	    t1=time()
            DNN.create_graph(a,b,c,d,e,f,g,train_index,test_index)
	    retorno=DNN.Training(modelo,0)
	    t2=time()
            Accuracy_kfold.append(retorno[0])
	    F1_Score_kfold.append(retorno[1])
	    Time_kfold.append((t2-t1)/60)
            kfold=kfold+1
            #DNN.Test()
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
        modelo = modelo+1
        print ('##################################################################################')
        file.write('##################################################################################\n')
    best=np.argmax(Estadisticos_F1_Score_Mean)
    print ('El mejor modelo es: {!s}'.format(best+1))
    file.write('El mejor modelo es: {!s}\n'.format(best+1))
    print ('epochs: {!s}'.format(BM[best][0]))
    file.write('epochs: {!s}\n'.format(BM[best][0]))
    print ('num_neurons: {!s}'.format(BM[best][1]))
    file.write('num_neurons: {!s}\n'.format(BM[best][1]))
    print ('learning_rate: {!s}'.format(BM[best][2]))
    file.write('learning_rate: {!s}\n'.format(BM[best][2]))
    print ('batch_size: {!s}'.format(BM[best][3]))
    file.write('batch_size: {!s}\n'.format(BM[best][3]))
    print ('layers: {!s}'.format(BM[best][4]))
    file.write('layers: {!s}\n'.format(BM[best][4]))
    print ('drop_out: {!s}'.format(BM[best][5]))
    file.write('drop_out: {!s}\n'.format(BM[best][5]))
    print ('optimizador: {!s}'.format(BM[best][6]))
    file.write('optimizador: {!s}\n'.format(BM[best][6]))
    print ("F1_Score_Mean: {!s}".format(Estadisticos_F1_Score_Mean[best])+" F1_Score_Std: {!s}".format(Estadisticos_F1_Score_Std[best]))
    file.write("F1_Score_Mean: {!s}".format(Estadisticos_F1_Score_Mean[best])+" F1_Score_Std: {!s}\n".format(Estadisticos_F1_Score_Std[best]))
    print ("Time_Mean: {!s}".format(Estadisticos_Time_Mean[best])+" Time_Std: {!s}".format(Estadisticos_Time_Std[best]))
    file.write("Time_Mean: {!s}".format(Estadisticos_Time_Mean[best])+" Time_Std: {!s}\n".format(Estadisticos_Time_Std[best]))
    DNN.create_graph(BM[best][0],BM[best][1],BM[best][2],BM[best][3],BM[best][4],BM[best][5],BM[best][6],None,None)
    DNN.Training(best+1,1)
    file.close()
