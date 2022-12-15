
#importing the libraries
# !pip install --quiet tf2_gnn
# !pip install --quiet networkx

import random
import math
import networkx as nk
import matplotlib.pyplot as plot
from matplotlib import cm
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tf2_gnn.layers.gnn import GNN, GNNInput
import tensorflow as tf
from tensorflow.math import segment_mean
from tensorflow import keras
from tensorflow.keras import Input, Model 
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
import pandas as pd




def load(file):
#reading the contents of the file
    f= open(file, 'r')     
#splitting the file by delimiter $$$$  in the file thus splitting each molecule in the samples array        
    molecules = f.read().split('$$$$')
    
#method to read each molecule configuration
#m  represents one molecule
    def analyzing_sample(m):
        lines = m.splitlines()
        links = []
        nodes = []
        label = 0
#for loop over each line
        for l in lines:
#Labels are assigned as 1, 0 if there are present as 1.0 and -1.0 in the dataset
            if l.strip() == '1.0':
                label = 1
            if l.strip() == '-1.0':
                label = 0
            if l.startswith('    '):
                feature = l.split()
                node = feature[3]
                nodes.append(node)
            elif l.startswith(' '):
                lnk = l.split()
                if int(lnk[0]) - 1 < len(nodes):
                    links.append(( int(lnk[0])-1, int(lnk[1])-1, ))                   
        return nodes, np.array(links), label 
    return [analyzing_sample(m) for m in tqdm(molecules) if len(m[0]) > 0]



#loading the training dataset
training_dataset = load('train.sdf')

"""#Plotting the Graph of one Molecule"""

c = cm.rainbow(np.linspace(0, 1, 50))
#plotting the graph by providing one molecule set
def plotting(set):
    graph=nk.Graph() 
    nodes = set[0] 
    edges = set[1] 
    labels={} 
    color_nodes=[] 
    for i,n in enumerate(nodes):
      graph.add_node(i) 
      labels[i]=n  
      color_nodes.append(c[hash(n)%len(c)]) 
    for e in edges:
        graph.add_edge(e[0], e[1]) 
    nk.draw(graph, labels=labels, with_labels = True, node_color = color_nodes)
    plot.show()
    return graph

plot.clf()
plotting(training_dataset[2])

"""#Preprocessing by tokenzing the data"""

#splitting the training dataset into train and validation set by 80 to 20 
train, validation = train_test_split(training_dataset, test_size=0.2,)
train, test = train_test_split(train, test_size=0.2,)

#Preprocessing
max_vocab = 500
max_len = 100
# build vocabulary from training set
all_nodes = [s[0] for s in train]
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(all_nodes)
random.seed(10)

def create_batch(data):
    nodes = [i[0] for i in data]  
#tokenizing the sample nodes                   
    nodes = tokenizer.texts_to_sequences(nodes)   
    nodes = pad_sequences(nodes, padding='post')  
    max_nodes_len = np.shape(nodes)[1]  
    edges = [s[1]+i*max_nodes_len for i,s in enumerate(data)] 
    edges = [e for e in edges if len(e) > 0]
    node_to_graph = [[i]*max_nodes_len for i in range(len(data))]  
    all_nodes = np.reshape(nodes, -1)  
    all_edges = np.concatenate(edges)         
    node_to_graph = np.reshape(node_to_graph, -1)
    return { 'data': all_nodes, 'edges': all_edges, 'node2grah': node_to_graph, }, np.array([s[2] for s in data]) 


#creating a batch 
def create(train, batch_size=16, repeat=False, shuffle=True):
    while True:   
        train = list(train) 
        if shuffle:   
            random.shuffle(train) 
        l = len(train)  
        for i in range(0, l, batch_size):  
            data = train[i:min(i + batch_size, l)] 
            yield create_batch(data)  
        if not repeat:  
            break

            
# displaying one batch:
for training in create(train, batch_size=4):
    for k,v in training[0].items():
        print(k)
        print(v)
        pass
    print('label', training[1])
    break

"""#Trial-1 using RGAT

"""

#Setting the parameters for GNN
dataset = keras.Input(batch_shape=(None,))                    
edge = keras.Input(batch_shape=(None, 2), dtype=tf.int32)    
node2graph = keras.Input(batch_shape=(None,), dtype=tf.int32) 
embeded = Embedding(tokenizer.num_words, 100)(dataset)    
num_graph = tf.reduce_max(node2graph)+1  


#The above inputs are passed to the GNN Input layer 
input = GNNInput( node_features=embeded, adjacency_lists=(edge,), node_to_graph_map=node2graph, num_graphs=num_graph,)


#initially getting all the default parameters
parameters = GNN.get_default_hyperparameters()
parameters["hidden_dim"] = 32 
#passing RGAT for GNN Parameters
parameters["message_calculation_class"] = 'RGAT'
parameters["num_heads"]=32
gnn_layer = GNN(parameters)  
output = gnn_layer(input) 

print('gnn_out', output)           

average = segment_mean( data=output, segment_ids=node2graph)   

print('mean:', print)
fc1 = Dense(8,activation='relu')(average) 
pred = Dense(1, activation='sigmoid')(average)  
print('pred:', pred)
model1 = Model( inputs={'data': dataset, 'edges': edge, 'node2grah': node2graph }, outputs=pred )

#printing the model summary
model1.summary()

#Training the model1
model1.compile( loss='BinaryCrossentropy', metrics=['AUC'])
batch_size = 32
batches = math.ceil(len(train) / batch_size)
validation_batch = math.ceil(len(validation) / batch_size)

model1.fit(create(train, batch_size=batch_size, repeat=True),
           steps_per_epoch=batches,
           epochs=40,
           validation_data=create(
           validation, batch_size=16, repeat=True),
           validation_steps=validation_batch)

y_test = []
for i in test:
  y_test.append(list(i)[2])
#make prediction on test data by using the trained model 
y_pred = model1.predict(create(test, batch_size=16, shuffle=False))
pred = []
for i in y_pred:
  if i[0] >= 0.5:
      pred.append(1)
  else:
      pred.append(0)


 #Accuracy of Trail-1
print("Accuracy of Trail-1:", accuracy_score(y_test,pred))
accuracy_score(y_test,pred)

#Balanced Accuracy of Trail-1
print("Balanced Accuracy of Trail-1:", balanced_accuracy_score(y_test,pred))

"""# Trail-2 using RGCN"""

#Setting the parameters for GNN
dataset = keras.Input(batch_shape=(None,))                    
edge = keras.Input(batch_shape=(None, 2), dtype=tf.int32)    
node2graph = keras.Input(batch_shape=(None,), dtype=tf.int32) 
embeded = Embedding(tokenizer.num_words, 100)(dataset)    
num_graph = tf.reduce_max(node2graph)+1  


#The above inputs are passed to the GNN Input layer  
input = GNNInput( node_features=embeded, adjacency_lists=(edge,), node_to_graph_map=node2graph, num_graphs=num_graph)


#initially getting all the default parameters
parameters = GNN.get_default_hyperparameters()
parameters["hidden_dim"] = 32 
#passing RGCN for GNN Parameters
parameters["message_calculation_class"] = 'RGCN'
parameters["num_edge_MLP_hidden_layers"] = 16
gnn_layer = GNN(parameters)  
output = gnn_layer(input) 

print('gnn_out', output)           

average = segment_mean( data=output, segment_ids=node2graph)   

print('mean:', average)
fc1 = Dense(8,activation='relu')(average) 
pred = Dense(1, activation='sigmoid')(average)  
print('pred:', pred)
model2 = Model( inputs={ 'data': dataset, 'edges': edge, 'node2grah': node2graph}, outputs=pred )

#printing the model summary
model2.summary()

#Training the model2
model2.compile( loss='BinaryCrossentropy', metrics=['AUC'])
batch_size = 32
batches = math.ceil(len(train) / batch_size) 
validation_batch = math.ceil(len(validation) / batch_size) 
model2.fit(create(train, batch_size=batch_size, repeat=True),
          steps_per_epoch=batches,
          epochs=40,
          validation_data=create(validation, batch_size=16, repeat=True),
          validation_steps=validation_batch)

y_test = []
for i in test:
  y_test.append(list(i)[2])
#make prediction on test data by using the trained model 
y_pred = model2.predict(create(test, batch_size=16, shuffle=False))
pred = []
for i in y_pred:
  if i[0] >= 0.5:
      pred.append(1)
  else:
      pred.append(0)

 #Accuracy of Trail-2
print("Accuracy of Trail-2:", accuracy_score(y_test,pred))

#Balanced Accuracy of Trail-2
print("Balanced Accuracy of Trail-2:", balanced_accuracy_score(y_test,pred))

"""#Balancing the Data"""

#passing the training dataset to a dataframe
data = pd.DataFrame(train, columns=['Nodes','Edges','Labels']) 
#data with Label 0
label_0 = data[data['Labels']==0]
#data with Label 1
label_1 = data[data['Labels']==1]  

print('Number of labels before Upsampling:')
print("Data with Label-0:", label_0.shape)
print("Data with Label-1:", label_1 .shape, '\n')

# Upsampling the data with Label 1
upsampled_data = resample(label_1, replace=True, n_samples=len(label_0), random_state=42)               
upsampled = pd.concat([label_0, upsampled_data])

print('Number of labels after Upsampling:')
upsampled['Labels'].value_counts()

#converting the dataframe of data to back to numpy array
train= upsampled.to_numpy()

"""#Trial-3 using RGAT on Balanced Data"""

#Setting the parameters for GNN
dataset = keras.Input(batch_shape=(None,))                    
edge = keras.Input(batch_shape=(None, 2), dtype=tf.int32)    
node2graph = keras.Input(batch_shape=(None,), dtype=tf.int32) 
embeded = Embedding(tokenizer.num_words, 100)(dataset)    
num_graph = tf.reduce_max(node2graph)+1  
#The above inputs are passed to the GNN Input layer 
input = GNNInput(node_features=embeded, adjacency_lists=(edge,), node_to_graph_map=node2graph, num_graphs=num_graph)


#initially getting all the default parameters
parameters = GNN.get_default_hyperparameters()
parameters["hidden_dim"] = 32 
#passing RGAT for GNN Parameters
parameters["message_calculation_class"] = 'RGAT'
parameters["num_heads"]=32
gnn_layer = GNN(parameters)  
output = gnn_layer(input) 
print('gnn_out', output)           
average = segment_mean( data=output, segment_ids=node2graph )   
print('mean:', average)
fc1 = Dense(8, activation='relu')(average) 
pred = Dense(1, activation='sigmoid')(average)  
print('pred:', pred)
model_bal_1 = Model(inputs={ 'data': dataset, 'edges': edge, 'node2grah': node2graph}, outputs=pred)


#printing the model summary
model_bal_1.summary()

#Training the model
model_bal_1.compile( loss='BinaryCrossentropy', metrics=['AUC'])
batch_size = 32
batches = math.ceil(len(train) / batch_size)
validation_batch = math.ceil(len(validation) / batch_size)

model_bal_1.fit(create(train, batch_size=batch_size, repeat=True),
           steps_per_epoch=batches,
           epochs=40,
           validation_data=create(
           validation, batch_size=16, repeat=True),
           validation_steps=validation_batch)

y_test = []
for i in test:
  y_test.append(list(i)[2])
#make prediction on test data by using the trained model 
y_pred = model_bal_1.predict(create(test, batch_size=16, shuffle=False))

pred = []
for i in y_pred:
  if i[0] >= 0.5:
      pred.append(1)
  else:
      pred.append(0)


#Accuracy of Trail-3
print("Accuracy of Trail-3:", accuracy_score(y_test,pred))


#Balanced Accuracy of Trail-3
print("Balanced Accuracy of Trail-3:", balanced_accuracy_score(y_test,pred))

"""#Trial-4 using RGCN on Balaned Data"""

dataset = keras.Input(batch_shape=(None,))                    
edge = keras.Input(batch_shape=(None, 2), dtype=tf.int32)    
node2graph = keras.Input(batch_shape=(None,), dtype=tf.int32) 
embeded = Embedding(tokenizer.num_words, 50)(dataset)    
num_graph = tf.reduce_max(node2graph)+1  


#The above inputs are passed to the GNN Input layer 
gnn_input = GNNInput(node_features=embeded, adjacency_lists=(edge,), node_to_graph_map=node2graph, num_graphs=num_graph)


#initially getting all the default parameters
parameters = GNN.get_default_hyperparameters()
parameters["hidden_dim"] = 64
#passing RGCN for GNN Parameters
parameters["message_calculation_class"] = 'RGCN'
gnn_layer = GNN(parameters)  
gnn_out = gnn_layer(gnn_input) 

print('gnn_out', gnn_out)           

average = segment_mean(data=gnn_out, segment_ids=node2graph )  

print('mean:', average)
fc1 = Dense(8,activation='relu')(average) 
pred = Dense(1, activation='sigmoid')(average)  
print('pred:', pred)
model_bal_2 = Model(inputs={'data': dataset, 'edges': edge, 'node2grah': node2graph }, outputs=pred)

#printing summary of the model
model_bal_2.summary()

#Training the model
model_bal_2.compile( loss='BinaryCrossentropy', metrics=['AUC'])
batch_size = 32
batches = math.ceil(len(train) / batch_size) 
validation_batch = math.ceil(len(validation) / batch_size) 
model_bal_2.fit(create(train, batch_size=batch_size, repeat=True),
          steps_per_epoch=batches,
          epochs=20,
          validation_data=create(validation, batch_size=16, repeat=True),
          validation_steps=validation_batch)

y_test = []
for i in test:
  y_test.append(list(i)[2])
#make prediction on test data by using the trained model 
y_pred = model_bal_2.predict(create(test, batch_size=16, shuffle=False))

pred = []
for i in y_pred:
  if i[0] >= 0.5:
      pred.append(1)
  else:
      pred.append(0)


#Accuracy of Trail-4
print("Accuracy of Trail-4:", accuracy_score(y_test,pred))


#Balanced Accuracy of Trail-4
print("Balanced Accuracy of Trail-4:", balanced_accuracy_score(y_test,pred))