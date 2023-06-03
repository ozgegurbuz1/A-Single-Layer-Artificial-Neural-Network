# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:03:29 2023

@author: ÖZGE
"""

import numpy as np

class Neuron:
    
    # __init__ ile başlangıçtaki oluşruduğumuz nesneye ağırlık ve bias atamalarını yapıyoruz. 
    def __init__ (self ,weight, bias):
        self.weight=weight
        self.bias=bias
        
        
    # elde edilen net değerimizi sigmoid aktivasyon fonksiyonuna sokuyoruz    
    def sigmoid(self,x):
        return 1/(1+ np.exp(-x))
    
    # ileri besleme ile net değerimizi elde ediyoruz. ağıırlık ve giriş verileri vektörlerinin matris çarpımı yapılır. ve biasla toplanır.
    #ileri besleme aşamasından geçip f(net) değeri elde edildikten sonra siggmoid fonksiyonuna gönderilir ve standartlaştırılmış 1 - 0 değerleri elde edilir.
    def feedForward(self,data):
        sumResult=np.dot(data,self.weight)+ self.bias
        return self.sigmoid(sumResult)
    
                        
x=[2,5,4]  #giriş vektörü
w=[1,3,6]  #ağırlık vektörü
b=7        #bias değeri

#Oluşturduğumuz class'ı kullanarak ağırlık ve bias değerlerini nöronumuza atıyoruz.
neuron=Neuron(w,b)
result=neuron.feedForward(x)
print(result) 
