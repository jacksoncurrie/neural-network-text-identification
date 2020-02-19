import numpy as np
import matplotlib.pyplot as plt
from ANN import ANN

# network setup 
inode = 784  # 28*28 matrix of values for the character
hnode = 100
onode = 10  # the 0-9 eg 10 possible output numbers
lr = 0.2  #learning rate

#create the neural network object
# you must have the ANN.py code
ann = ANN(inode, hnode, onode, lr)

# set the training file name
dataFile = open('data\small_handwriting100.csv')
dataList = dataFile.readlines()
dataFile.close()

for record in dataList:
    recordx = record.split(',')
    inputT = (np.asfarray(recordx[1:]) *(1/255))
    train = np.zeros(onode) +0.1
    train[int(recordx[0])] = 0.99
    ann.trainNet(inputT, train)

match = 0
no_match = 0

for record in dataList:
    recordz = record.split(',')
    labelz = int(recordz[0])
    inputz = (np.asfarray(recordz[1:]) *(1/255))
    outputz = ann.testNet(inputz)
    max_value = np.argmax(outputz)
    if max_value == labelz:
        match = match + 1
    else:
        no_match = no_match + 1
        print ("Success Rate % = ",(float(match)/float(match + no_match))*100)
        
    