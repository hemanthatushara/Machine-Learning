#Hemantha Govindu
#1001531660

import sys
import numpy as np
import pandas as pd
from decimal import Decimal
import math
import statistics as stat
argumentList = sys.argv

def linear_regression (training_file, test_file,degree, lambda_val  ):
     dataset = np.genfromtxt(training_file, dtype=np.float32)
     #print (dataset)
     dataset_1 = np.genfromtxt(test_file, dtype = np.float32)
     class_train = dataset[:,-1]
     class_test = dataset_1[:,-1]

     attr_train = dataset[:,:-1]
     attr_test = dataset_1[:,:-1]
     #print(attr_train.shape[0])
     #k = 1
     phi_matrix = np.ones([attr_train.shape[0],(attr_train.shape[1]*int(degree))+1])
     #print(phi_matrix)
     #print(phi_matrix)
     for i in range(0,phi_matrix.shape[0]):
         k=1
         for j in range (0, attr_train.shape[1]):
             for d in range (1, int(degree)+1):
                     #print ('goes here')
                     phi_matrix[i][k] = np.power(attr_train[i][j],d)
                     k += 1
     #print(phi_matrix)
     phi_phi = np.matmul(np.transpose(phi_matrix),phi_matrix)
     #print(phi_phi)
     iden=np.multiply( np.identity(phi_phi.shape[0]),int(lambda_val))
     inverse = np.linalg.pinv(iden+phi_phi)
     target = np.matmul(np.transpose(phi_matrix),class_train)
     #print(target,        inverse)

     w = np.matmul(inverse, target)
     #print(len(w))
     #print(w.shape[0])
     for i in range(0, len(w)):
         print('w%d = %.4f'%(i, w[i]))

     phi_matrix1 = np.ones([attr_test.shape[0],(attr_test.shape[1])*int(degree)+1])

     for i in range(0,phi_matrix1.shape[0]):
         k=1
         for j in range (0, attr_test.shape[1]):
              for d in range (1, int(degree)+1):
                      #print ('goes here')
                      phi_matrix1[i][k] = np.power(attr_test[i][j],d)
                      k += 1
     for row in range (0, attr_test.shape[0]):
        res = np.dot(w,phi_matrix1[row])

        print('ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f' %(row+1, res, class_test[row], np.square(res - class_test[row])))


linear_regression (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
