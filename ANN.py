#
#
# MYNN is the program to run
# This code is only a loadable module which is called 
# by the program MYNN.py
#
#

import numpy as np
import math

class ANN:

    def __init__ (self, inode,hnode,onode,lr):
#set local variables
        self.inode = inode
        self.hnode = hnode
        self.onode = onode
        self.lr = lr

# standard deviation used for this code
        stdev = 0.3333

#mean is the reciprocal of the square root of the total nodes
        # pow expectes two numbers num,num  first is number to be operated on
        # second is the power you are raising the number by
        mean = (1/(pow((inode + hnode + onode), 0.5)))
        
    
# generate both weighting matrices
# input to hidden layer matrix
#    self.wtgih = np.random.normal(mean, 0.3333, [hnode, inode])
        self.wtgih = np.random.normal(mean, stdev, [hnode, inode])
        print ("WEIGHTING MATRIX FOR wtgih ")
        print (self.wtgih)
        print

# hidden to output layer matrix
#    self.wtgho = np.random.normal(mean, 0.3333, [onode, hnode])
        self.wtgho = np.random.normal(mean, stdev, [onode, hnode])
        print ("WEIGHTING MATRIX FOR wtgho ")
        print (self.wtgho)
        print
   
####################

    def testNet(self, input):
# convert input tuple to an array
        input = np.array(input, ndmin=2).T

# multiply input by wtgih
        hInput = np.dot(self.wtgih, input)

# sigmiod adjustment
        hOutput = 1/(1 + np.exp(-hInput))

# multiply hidden layer output by wtgho
        oInput = np.dot(self.wtgho, hOutput)

# sigmoid adjustment
        oOutput = 1/(1 + np.exp(-oInput))
        
        return oOutput

####################### 

    def trainNet(self, inputT, train):
# this module depends on the values, arrays and matrices
# created when the init module is run

# create the arrays from the list arguments
        self.inputT = np.array(inputT, ndmin=2).T
        self.train = np.array(train, ndmin=2).T

# multiply inputT array by wtgih
        self. hInputT = np.dot(self.wtgih, self.inputT)

# sigmoid adjustment
        self.hOutputT = 1/(1 + np.exp(-self.hInputT))

# multiply hidden layer output by wtgho
        self.oInputT = np.dot(self.wtgho, self.hOutputT)

# sigmoid adjustment
        self.oOutputT = 1/(1 + np.exp(-self.oInputT))

# calculate output errors
        self.eOutput = self.train - self.oOutputT

#calculate hidden layer error array
        self.hError = np.dot(self.wtgho.T, self.eOutput)

# update weight matrix wtgho
        self.wtgho += self.lr * np.dot((self.eOutput * self.oOutputT * (1 - self.oOutputT)), self.hOutputT.T)

# update weight matrix
        self.wtgih += self.lr * np.dot((self.hError * self.hOutputT * (1 - self.hOutputT)), self.inputT.T)

        print ("UPDATED WEIGHTING MATRIX FOR wtgih ")
        print (self.wtgih)
        print
        print ("UPDATED WEIGHTING MATRIX FOR wtgho ")
        print (self.wtgho)
        print




