import numpy as np


class Model(object):
    
    def __init__(self):
        
        self.weights = [np.zeros(shape=(24, 16)), 
                        np.zeros(shape=(16, 16)), 
                        np.zeros(shape=(16, 4))]
        
    def predict(self, input):
        
        output = np.expand_dims(input.flatten(), 0)
        output = output / np.linalg.norm(output)
        for layer in self.weights:
            output = np.dot(output, layer)
        return output[0]
    
    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights
        