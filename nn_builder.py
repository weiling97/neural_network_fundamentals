import numpy as np

class NNBuilder:
    
    ## initialise parameters
    @classmethod
    def initialise_parameters(self):
        # np.random.seed(42)
        parameters = {}

        parameters['A1'] = np.random.randn(4,2) * 0.01
        parameters['B1'] = np.zeros((4,1))
        parameters['A2'] = np.random.randn(1,4) * 0.01
        parameters['B2'] = np.zeros((1,1))

        return parameters
    
    ## forward propagation
    @classmethod
    def forward_propagation(self, X, Y, parameters):

        Y1 = self.activation_function_relu(np.dot(parameters['A1'], X)) + parameters['B1']
        Y2 = self.activation_function_sigmoid(np.dot(parameters['A2'], Y1)) + parameters['B2']

        return Y1, Y2

    @classmethod
    def activation_function_relu(self, z):
        
        return np.maximum(0, z)

    @classmethod
    def activation_function_sigmoid(self, z):
        
        return 1/(1+np.exp(-z))
    
    ## cross entropy loss
    @classmethod
    def cross_entropy_loss(self, Y2, Y):
        n = Y.shape[1]

        loss = -np.sum(Y*np.log(Y2) + (1-Y)*np.log(1-Y2))/n
        
        return loss
    
    ## back propagation
    @classmethod
    def back_propagation(self, X, Y, parameters, Y1, Y2):
        n = X.shape[1]

        #gradient
        dZ2 = Y2 - Y
        dY1 = np.dot(parameters['A2'].T, dZ2)
        dZ1 = dY1 * self.sigmoid_activation_function_derivative(Y1)
        dY0 = np.dot(parameters['A1'].T, dZ1)

        #parameter gradient
        dA1 = np.dot(dZ2, Y1.T)/n
        dB1 = np.sum(dZ2, axis=1, keepdims=True)/n
        dA0 = np.dot(dZ1, X.T)/n
        dB0 = np.sum(dZ1, axis=1, keepdims=True)/n

        gradients = {
            'dA2': dA1,
            'dB2': dB1,
            'dA1': dA0,
            'dB1': dB0
        }

        return gradients

    @classmethod
    def sigmoid_activation_function_derivative(self,sig):

        return sig * (1-sig)
    
    ## update parameters
    @classmethod
    def update_parameters(self, parameters, gradients, learning_rate=0.1):
        parameters['A2'] -= learning_rate * gradients['dA2']
        parameters['B2'] -= learning_rate * gradients['dB2']
        parameters['A1'] -= learning_rate * gradients['dA1']
        parameters['B1'] -= learning_rate * gradients['dB1']
        
        return parameters
    

if __name__ == '__main__':
    test = NNBuilder.initialise_parameters()
    print(test)