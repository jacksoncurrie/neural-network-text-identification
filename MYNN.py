#
#
# MYNN.py
#
#

import numpy as np
import matplotlib.pyplot as plt
from ANN import ANN

# network setup 
inode = 784  # 28*28 matrix of values for the character
#hnode = 100
hnode = 392  # nodes in the  hidden layer  784/2
onode = 10  # the 0-9 eg 10 possible output numbers
lr = 0.2  #learning rate

#create the neural object
ann = ANN(inode, hnode, onode, lr)

# set the training file name
dataFile = open ('data\small_handwriting100.csv')
dataList = dataFile.readlines()
dataFile.close()

#recordx = np.record.split(',')
recordx = dataList[0].split(',')


#adjustRecord0 = (np.asfarray(recordx[1:]) / (255.0 * 0.99)) + 0.01
adjustRecord1 = (np.asfarray(recordx[1:]) * (1/255))

#inputT = (np.asfarray(recordx[1:]) /(255* 0.99)) + 1
inputT = (np.asfarray(recordx[1:]) * (1/255))

train = np.zeros(onode) + 0.01
train[int(recordx[0])] = 0.99
ann.trainNet(inputT, train)


# possible problem here with the final value eg if record0 is between 255
# and 252.46  then the result is larger than 1.
# aren't the matrix values are supposed to be between 0 and 1

# 255*.99 = 252.45
# (255*.99) +.01 = 252.45

# perhaps a better math model is multiply by 1/255  = 0.0039215686274509803921568627451‬
# so if input value is 255 then 255 * (1/255) = 1
# 254 * (1/255) = 0.99607843  which is < 1