#Hemantha Govindu
#1001531660

import sys
import os
import numpy as np
import math


def k_means(data_file, k , initialization):
    data_points = np.loadtxt(data_file)
    #print(data_points)
    clusters = []
    j = 1
    for i in range(0,len(data_file)):
        if j>3:
            j =1
        clusters.append(j)
        j +=1
    print(clusters)


data_file = sys.argv[1]
k = sys.argv[2]
initialization = sys.argv[3]
k_means(data_file,int(k), initialization)
