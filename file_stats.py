#Hemantha Govindu
#1001531660
import numpy as np
import sys
def file_stats(pathname):
    
    #pathname = input("Enter pathname:")
    my_file = open(pathname,'r')
    num = my_file.readlines()
    
    for i in range(0,len(num)):
        
        num[i]=num[i].strip()
        #print(num[i])
       # num[i]=float(num[i])
        
    arr = np.array(num)
    arr = arr.astype(np.float)
    #print(arr)
    avg = np.mean(arr)
    std = (arr-avg)**2
    stdev =  (np.sum(std )/ (len(arr)-1))**(1/2)
    print("Average = ",avg)
    print("Standard deviation =", stdev)
    stdev=0
    return avg,stdev

        

(avg, stdev) = file_stats(sys.argv[1])