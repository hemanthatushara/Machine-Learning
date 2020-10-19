#Hemantha Govindu
#1001531660

import sys
import numpy as np
import pandas as pd
from decimal import Decimal
import math
import statistics as stat
argumentList = sys.argv 


def prob_class(find_class,total_classes):
  return np.divide(find_class,total_classes)

def gaussian(x,avg, sigma):
  
  a = np.multiply(((np.multiply(2, np.pi)**(1/2))), sigma)
  b = np.divide(1,a)
  c = (float(x-avg))**2
  d = math.exp(-(np.divide(c,(np.multiply(2,sigma**2)))))
  return  np.multiply(b,d)

def naive_bayes(training_file, test_file):
    
  dataset = np.genfromtxt(training_file)
  #print(dataset[:,-1])
  dataset_1 = np.genfromtxt(test_file)
  #print(dataset_1[:,-1])
  class_train = dataset[:,-1]
  class_test = dataset_1[:,-1]

  attr_train = dataset[:,:-1]
 
  attr_test = dataset_1[:,:-1]
  unique_class = np.unique(class_train)
  unique_class_test = np.unique(class_test)

  mean_list = []
  stdev_list = []
  #print(len(attr_train[0]))
   
  for i in range(0,len(unique_class)):
    some_class = np.where(class_train == unique_class[i])
    for k in range(0,len(some_class)):
      for j in range(0, len(attr_train[0])):
      
        mean =(np.mean(attr_train[some_class[k],j]))
        #mean_f= Decimal("% .2f" %mean)
        
        stdev=(stat.stdev(attr_train[some_class[k],j]))
      # stdev_f = Decimal("% .2f" %stdev)
        if stdev < 0.01:
          stdev = 0.01
        #print( unique_class[i] ,'=', attr_train[some_class,j])
        #print('class ', int( unique_class [i]), ', Attribute number', int(j+1), ', mean = ' , mean_f, ', std =', stdev_f)
        print ('Class %d, attribute %d, mean = %.2f, std = %.2f'%(int( unique_class [i]), int(j+1), mean,stdev))

        mean_list.append(mean)
        stdev_list.append(stdev)

  p_of_c_list = []
  p_of_x_list = []
  for i in range(0,len(unique_class)):
    find_class=np.asarray(np.where(class_train == unique_class[i]))
      
    fc = find_class.shape

    p_of_C=prob_class(fc[1],len(class_train))
    p_of_c_list.append(p_of_C)

    p_of_x = np.sum(p_of_c_list[i] )
    p_of_x_list.append(p_of_x)
   # print('probability of getting class', i+1,'is',p_of_C)
  
  p_x_given_C =1
  p_x_given_C_list = [] 
  max_val_list = []
  class_predicted = []
  accuracy_list = []

  l =0
  for i in range (0,len(class_test)):
    l=0
    for j in range (0, len(unique_class_test)):
      p_x_given_C =1
      for k in range (0, attr_test.shape[1]):  
         p_x_given_C *= gaussian(attr_test[i][k], mean_list[l], stdev_list[l])
         #print(p_x_given_C, l)
         l+=1 
      p_x_given_C_1 = np.multiply(p_x_given_C,p_of_c_list[j] )
      p_x_given_C_list.append(p_x_given_C_1)
         
    
    p_x_given_C_list[:]= [a/np.sum(p_x_given_C_list) for a in p_x_given_C_list ]
    max_val = max(p_x_given_C_list)
    c_l = p_x_given_C_list.index(max_val)
    class_predicted.append(c_l+1)
    
    max_val_list.append(max_val)
    p_x_given_C_list.clear()
    max_val =0
    if( class_predicted[i]==class_test[i]):
      accuracy = 1
      accuracy_list.append(accuracy )
    else:
      accuracy=0
      accuracy_list.append(accuracy)
    
    #print(max_val, c_l+1, accuracy_list[i])

  prob_list = []
  
  
  for h in range (0,len(class_test)):
    print('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f' %(h+1,class_predicted[h], max_val_list[h], class_test[h], accuracy_list[h]))
  
  print('Classification Accuracy = %6.4f' %(np.sum(accuracy_list)/len(accuracy_list)))
naive_bayes(sys.argv[1],sys.argv[2])

