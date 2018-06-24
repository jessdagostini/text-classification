# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 00:48:36 2018

@author: Jessica Dagostini
"""

# TensorFlow é a biblioteca responsável pela execução e montagem da rede neural
import tensorflow as tf
# Pandas é utilizado para manipular datasets
import pandas as pd
# Numpy é ótimo para utilização de multiplicação de matrizes
import numpy as np
# Counter é usado para guardar o vocabulario
from collections import Counter
import matplotlib as mp
import matplotlib.pyplot as plt

#INCIO -- Manipulação das informações para passar para a Rede Neural --

# Precisamos transoformar as palavras do vocabulario em números
# Transforma a palavra em números, de acordo com a quantidade de vezes que a palavra se repetiu
def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i
        
    return word2index

# Cria as camadas de entrada da rede neural
def text_to_vector(text):
    layer = np.zeros(total_words,dtype=float)
    for word in text.split(' '):
        layer[word2index[word.lower()]] += 1
        
    return layer

# Cria o vetor com as saidas esperadas
def category_to_vector(category):
    y = np.zeros((2),dtype=float)
    # Categoria 0 = avaliação negativa
    if category == 0:
        y[0] = 1.
    # Categoria 1 = avaliação positiva
    elif category == 1:
        y[1] = 1.
        
    return y

# Função para pegar os dados por batches
def get_batch(df,i,batch_size):
    batches = []
    results = []
    texts = df[0][i*batch_size:i*batch_size+batch_size]
    categories = df[1][i*batch_size:i*batch_size+batch_size]
    
    for text in texts:
        layer = text_to_vector(text) 
        batches.append(layer)
        
    for category in categories:
        y = category_to_vector(category)
        results.append(y)  
     
    return np.array(batches),np.array(results)

#FIM -- Manipulação das informações para passar para a Rede Neural --

#INICIO -- Rede Neural -- 

# O perceptron
def multilayer_perceptron(input_tensor, weights, biases):
    # Camada 1 com função de ativação sigmoid
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.elu(layer_1_addition)
    
    # Camada 2 com função de ativação Relu
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.elu(layer_2_addition)
    
    # Camda de saída
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition

#FIM -- Rede Neural --
    
#INICIO -- Definição dos parametros para a treinamento da rede neural -- 

# Definição do vocabulario
vocabulario = Counter()
# Datasets de avaliações
train = pd.read_csv('datasets/yelp_labelled.csv', sep="\t", header=None)
test = pd.read_csv('datasets/amazon_cells_labelled.csv', sep="\t", header=None)

# Criação do vocabulario
for text in train[0]:
    for word in text.split(' '):
        vocabulario[word.lower()]+=1

for text in test[0]:
    for word in text.split(' '):
        vocabulario[word.lower()]+=1

# Transformação do vocabulario de palavras em numeros    
word2index = get_word_2_index(vocabulario)
# Total de palavras do vocab
total_words = len(vocabulario)


# Parametros
learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]     # Taxa de aprendizagem
training_epochs = 300   # Quantidade de épocas para treinamento
batch_size = 350        # Tamanho do batch de palavras que será capturada por vez
display_step = 100      # Para mostrar o erro

# Parametros da rede neural
n_hidden_1 = 100      # Numero de neuronios na primeira camada
n_hidden_2 = 100       # Numero de neuronios na segunda camada
n_input = total_words # Numero de neuronios para a entrada
n_classes = 2         # Categorias: positiva(1) e negativa(0)
accuracies = []
l_rates = []

# Placeholders necessarios para o tensorflow
input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output")

# Definição dos pesos
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
# Definição dos biases
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

for learning_rate in learning_rates:
    print("Learning rate: " + str(learning_rate))
    # Montagem da rede neural
    prediction = multilayer_perceptron(input_tensor, weights, biases)
    
    # Definindo o erro e a otimização
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # Iniciando as variáveis
    init = tf.global_variables_initializer()
    
    #FIM -- Definição dos parametros para a treinamento da rede neural -- 
    
    # [NEW] Add ops to save and restore all the variables
    #saver = tf.train.Saver()
    
    #INICIO -- Treinando a rede -- 
    
    # Executando a rede neural
    with tf.Session() as sess:
        sess.run(init)
    
        # Itera pela quantidade de épocas definida
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(train)/batch_size)
            # Itera por todos os batches
            for i in range(total_batch):
                batch_x,batch_y = get_batch(train,i,batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                # Roda a rede neural, levando em consideração o erro e a otimização
                c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
                #print("C:", c)
                # Compute average loss
                # Media de custo
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch), "loss=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        
        #INICIO -- Testando o modelo para ver sua acurácia -- 
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Testando com o dataset da amazon
        total_test_data = len(test[1])
        batch_x_test,batch_y_test = get_batch(test,0,total_test_data)
        accur = accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test})
        print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))
        l_rates.append(learning_rate)
        accuracies.append(accur)
        
        x_10_texts,y_10_correct_labels = get_batch(test,0,10)
        classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: x_10_texts})
        print("Test with 10 texts")
        print("Predicted categories:", classification)
        print("Correct categories:  ", np.argmax(y_10_correct_labels, 1))
        
        print("\n")

plt.plot(l_rates, accuracies)
plt.grid(True)
plt.show()