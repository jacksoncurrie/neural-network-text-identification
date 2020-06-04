#read external csv formatted file and display the contents as an image eg what number is inside the file

import numpy as np
import matplotlib.pyplot as plt

# open the data file, assume it is in the same location as this fie
dataFile = open ('data/jackson_list_3.csv')
dataList = dataFile.readlines()
dataFile.close()

for i in range(7):

    # separate each of the data items by using the comma character as the delimiter
    record0 = dataList[i].split(',')
    
    # Read the actual value from the file and alter to be within the function range >0 and <1
    adjustRecord0 = (np.asfarray(record0[1:]) / (255.0 * 0.99)) + 0.01
    
    # arrange the data array into rows/colunm of 28*28
    imageArray = np.asfarray(record0[1:]).reshape((28,28))
    
    # set the colours to be used in the display
    plt.imshow(imageArray, cmap='Greys')
    plt.show()
