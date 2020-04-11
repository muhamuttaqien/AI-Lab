import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import skfuzzy as fuzzy

class Fuzzy:
    def __init__(self,interval):
        """
            interval for fuzzy input arange
            need in form of [min,max,span]
        """
        self.fuzzy=fuzzy
        assert len(interval)==3, "need in form of [min,max,span]"
        self.x=np.arange(interval[0], interval[1], interval[2])
        self.fuzzy_set={}
        self.n_fset=None

    def trinf(self,inputs,u_key):
        """
            inputs--> data to make a trangle 
            inputs[0]-->left value of triangle
            inputs[1]-->top of the traingle
            inputs[2]--> right value of triangle
        """
        assert len(inputs)==3, "need in form of [min,max,min]"
        self.fuzzy_set[u_key]=self.fuzzy.trimf(self.x,inputs)
        return self.fuzzy.trimf(self.x,inputs)

    def trapmf(self,inputs,u_key):
        """
            inputs--> data to make a trapezoide 
            inputs[0]-->left value of trapezoide 
            inputs[1]-->top left of the trapezoide 
            inputs[2]--> top right value of trapezoide
            inputs[3]-->right value of trapezoide 
        """
        assert len(inputs)==4, "need in form of [min,max,max,min]"
        self.fuzzy_set[u_key]=self.fuzzy.trapmf(self.x,inputs)
        return self.fuzzy.trapmf(self.x,inputs)

    def gaussmf(self,inputs,u_key):
        """
            inputs--> data to make a gaussian
            inputs[0]-->value of mean 
            inputs[1]-->value of the sigma 
        """
        assert len(inputs)==2, "need in form of [mean,sd]"
        self.fuzzy_set[u_key]=self.fuzzy.gaussmf(self.x,inputs[0],inputs[1])
        return self.fuzzy.gaussmf(self.x,inputs[0],inputs[1])

    def _plot(self,*args):
        """
            inputs--> data to plot as y
            base x is first class initialization time inputs
        """
        #assert len(*args)==len(self.x), "need in form of len(x)"
        c=["g","m","b","k"]
        plt.figure(figsize=(10,5))
        for ids,i in enumerate(args):
            plt.plot(self.x, i, 'b', linewidth=1.5,color=c[ids])
        plt.show()

    def _interp_membership(self,val,inputs):
        """
            val--> fuzzy set value 
            inputs--> real value to quantify by using the val or fuzzy set
            return ---> interprete membership value
        """
        assert len(val)==len(self.x), "need in form of len(x)"
        inputs=int(inputs)
        return self.fuzzy.interp_membership(self.x, val, inputs)