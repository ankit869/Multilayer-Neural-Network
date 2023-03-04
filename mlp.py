import numpy as np
import pandas as pd
import random

class Activation_Functions():
    def __init__(self):
        super(Activation_Functions,self).__init__()
        self.leaky_relu_fraction = 0.01
        self.elu_alpha = 0.1

    def sigmoid(self, z, derive=False):
        if (derive):
            return (z*(1-z))
        else:
            if (isinstance(z, np.ndarray)):
                return np.array([1/(1+np.exp(-i)) for i in z])
            else:
                return 1/(1+np.exp(-z))

    def relu(self, z, derive=False):
        if (derive):
            return np.where(z > 0, 1, 0)
        else:
            if (isinstance(z, np.ndarray)):
                return np.array([max(0, i) for i in z])
            else:
                return max(0, z)

    def tanh(self, z, derive=False):
        if (derive):
            if (isinstance(z, np.ndarray)):
                return np.array([(1-self.tanh(i)**2) for i in z])
            else:
                return 1-self.tanh(z)**2
        else:
            if (isinstance(z, np.ndarray)):
                return np.array([((np.exp(i)-np.exp(-i))/(np.exp(i)+np.exp(-i))) for i in z])
            else:
                return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

    def elu(self, z, derive=False):
        if (derive):
            return np.where(z > 0, 1, self.elu(z)+self.elu_alpha)
        else:
            return np.where(z > 0, z, self.elu_alpha*(np.exp(z)+self.elu_alpha))

    def leaky_relu(self, z, derive=False):
        if (derive):
            return np.where(z >= 0, 1, self.leaky_relu_fraction)
        else:
            if (isinstance(z, np.ndarray)):
                return np.array([max(self.leaky_relu_fraction*i, i) for i in z])
            else:
                return max(self.leaky_relu_fraction*z, z)

    def linear(self, z, derive=False):
        if (derive):
            return 1
        else:
            return z

class Losses:
    def __init__(self):
        super(Losses,self).__init__()

    def mse(self, y, p):
        return np.mean((y - p) ** 2)

    def binary_cross_entropy(self, y, p):
        return np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))


class instance_variables:
    def __init__(self):
        super(instance_variables,self).__init__()
        self.weights = []
        self.bias = []
        self.Vdb = []
        self.Vdw = []
        self.Mdw = []
        self.Mdb = []
        self.derivatives_w=[]
        self.derivatives_b=[]
        self.regularization = None
        self.activations = []
        self.dropout_nodes = []
        self.layers = []


class Weight_Initalizer(instance_variables):

    def __init__(self):
        super(Weight_Initalizer,self).__init__()
        self.weight_initializer = {
            'random_uniform': self.random_uniform,
            'random_normal': self.random_normal,
            'glorot_uniform': self.glorot_uniform,
            'glorot_normal': self.glorot_normal,
            'he_uniform': self.he_uniform,
            'he_normal': self.he_normal
        }

    def random_uniform(self, seed=None,args=dict()):
        minval = -0.05
        maxval = 0.05
        for key, value in args.items():
            if (key == 'minval'):
                minval = value
            elif (key == 'maxval'):
                maxval = value
            elif(key=='seed'):
                np.random.seed(seed)
                
        for i in range(len(self.layers)-1):
            self.weights[i] = np.random.uniform(minval, maxval, size=(
                self.layers[i+1]['nodes'], self.layers[i]['nodes']))

    def random_normal(self,args=dict()):
        for key, value in args.items():
            if(key=='seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            self.weights[i] = np.random.randn(self.layers[i+1]['nodes'], self.layers[i]['nodes'])

    def glorot_uniform(self,args=dict()):
        for key, value in args.items():
            if(key=='seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            limit = np.sqrt(6 / (self.layers[i+1]['nodes'] + self.layers[i]['nodes']))
            vals = np.random.uniform(-limit, limit,size=(self.layers[i+1]['nodes'], self.layers[i]['nodes']))
            self.weights[i] = vals

    def glorot_normal(self,args=dict()):
        for key, value in args.items():
            if(key=='seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            limit = np.sqrt(2 / (self.layers[i+1]['nodes'] + self.layers[i]['nodes']))
            vals = np.random.randn(self.layers[i+1]['nodes'], self.layers[i]['nodes'])*limit
            self.weights[i] = vals

    def he_uniform(self, seed=None,args=dict()):
        for key, value in args.items():
            if(key=='seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            limit = np.sqrt(6 / (self.layers[i]['nodes']))
            vals = np.random.uniform(-limit,limit,size=(self.layers[i+1]['nodes'], self.layers[i]['nodes']))
            self.weights[i] = vals

    def he_normal(self, args=dict()):
        for key, value in args.items():
            if(key=='seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            vals = np.random.randn(
                self.layers[i+1]['nodes'], self.layers[i]['nodes']) * np.sqrt(2/(self.layers[i]['nodes']))
            self.weights[i] = vals


class Optimizers(Weight_Initalizer):
    def __init__(self):
        super(Optimizers,self).__init__()
        self.epsilon = 1e-07
        self.momentum = 0.9
        self.beta = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999

    def get_gradients(self, layer):
        derivatives_w = None
        derivatives_b = None
        if(self.regularization=='L2_norm'):
            derivatives_w = self.derivatives_w[layer] + (2*self.penalty*self.weights[layer])
        elif(self.regularization=='L1_norm'):
            derivatives_w = self.derivatives_w[layer]+self.penalty
        else: 
            derivatives_w = self.derivatives_w[layer]

        derivatives_b = self.derivatives_b[layer+1].flatten()
        return derivatives_w, derivatives_b

    def gradient_descent(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.weights[i] -= dw * learningRate
            self.bias[i+1] -= db * learningRate

    def Momentum(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Vdw[i] = (self.momentum*self.Vdw[i]) + (dw*learningRate)
            self.Vdb[i+1] = (self.momentum*self.Vdb[i+1]) + (db*learningRate)
            self.weights[i] -= self.Vdw[i]
            self.bias[i+1] -= self.Vdb[i+1]

    def AdaGrad(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Vdw[i] = self.Vdw[i]+(dw**2)
            self.Vdb[i+1] = self.Vdb[i+1]+(db**2)
            self.weights[i] -= (learningRate/np.sqrt(self.Vdw[i]+self.epsilon))*dw
            self.bias[i+1] -= (learningRate/np.sqrt(self.Vdb[i+1]+self.epsilon))*db

    def RMSprop(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Vdw[i] = self.beta*self.Vdw[i]+(1-self.beta)*(dw**2)
            self.Vdb[i+1] = self.beta*self.Vdb[i+1]+(1-self.beta)*(db**2)
            self.weights[i] -= (learningRate/np.sqrt(self.Vdw[i]+self.epsilon))*dw
            self.bias[i+1] -= (learningRate/np.sqrt(self.Vdb[i+1]+self.epsilon))*db

    def Adam(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Mdw[i] = self.beta1*self.Mdw[i]+(1-self.beta1)*dw
            self.Vdw[i] = self.beta2*self.Vdw[i]+(1-self.beta2)*(dw**2)
            m_dw = self.Mdw[i]/(1-self.beta1)
            v_dw = self.Vdw[i]/(1-self.beta2)
            self.weights[i] -= (learningRate/np.sqrt(v_dw+self.epsilon))*m_dw

            self.Mdb[i+1] = self.beta1*self.Mdb[i+1]+(1-self.beta1)*db
            self.Vdb[i+1] = self.beta2*self.Vdb[i+1]+(1-self.beta2)*(db**2)
            m_db = self.Mdb[i+1]/(1-self.beta1)
            v_db = self.Vdb[i+1]/(1-self.beta2)
            self.bias[i+1] -= (learningRate/np.sqrt(v_db+self.epsilon))*m_db

class MultiLayerNeuralNetwork(Activation_Functions, Losses, Optimizers):
    def __init__(self):
        super(MultiLayerNeuralNetwork,self).__init__()
        self.history = {'Losses': [], 'Weights': [],
                        'Bias': [], 'Activations': []}
        self.loss_functions = {
            "mse": self.mse,
            "binary_cross_entropy": self.binary_cross_entropy
        }
        self.optimizer = 'RMSprop'
        self.optimizer_function = {
            'momentum': self.Momentum,
            'gradient_descent': self.gradient_descent,
            'AdaGrad': self.AdaGrad,
            'RMSprop': self.RMSprop,
            'Adam': self.Adam
        }
        self.activation_functions = {
            "sigmoid": self.sigmoid,
            "relu": self.relu,
            "leaky_relu": self.leaky_relu,
            "elu": self.elu,
            "tanh": self.tanh,
            "linear": self.linear
        }

    def add_layer(self, nodes=3, activation_function='linear', input_layer=False, output_layer=False, dropouts=False, dropout_fraction=None, **kwargs):
        if (input_layer is not False):
            self.n_inputs = nodes
            self.layers.append(
                {'nodes': nodes, 'activation_function': 'linear', 'dropouts': False})
        elif (output_layer is not False):
            self.n_outputs = nodes
            self.layers.append(
                {'nodes': nodes, 'activation_function': activation_function, 'dropouts': False})
        else:
            self.layers.append({'nodes': nodes, 'activation_function': activation_function,
                               'dropouts': dropouts, 'dropout_fraction': dropout_fraction})

    def compile_model(self, loss_function='mse', weight_initializer='glorot_uniform', **kwargs):
        self.loss_func = loss_function
        for i in range(len(self.layers)):
            self.activations.append(np.zeros(self.layers[i]['nodes']))
            self.bias.append(np.zeros(self.layers[i]['nodes']))
            self.Vdb.append(np.zeros(self.layers[i]['nodes']))
            self.Mdb.append(np.zeros(self.layers[i]['nodes']))
            self.dropout_nodes.append(
                np.zeros(self.layers[i]['nodes'], dtype=bool))
            self.derivatives_b.append(np.zeros(self.layers[i]['nodes']))

            if (self.layers[i]['dropouts'] == True):
                self.add_dropouts(i, self.layers[i]['dropout_fraction'])

        for i in range(len(self.layers)-1):
            self.Vdw.append(np.zeros((self.layers[i+1]['nodes'], self.layers[i]['nodes'])))
            self.Mdw.append(np.zeros((self.layers[i+1]['nodes'], self.layers[i]['nodes'])))
            self.weights.append(np.random.rand(self.layers[i+1]['nodes'], self.layers[i]['nodes']))
            self.derivatives_w.append(np.zeros((self.layers[i+1]['nodes'], self.layers[i]['nodes'])))

        self.weight_initializer[weight_initializer](kwargs)

        for key, value in kwargs.items():
            if (key == 'leaky_relu_fraction'):
                self.leaky_relu_fraction = value
            elif (key == 'elu_alpha'):
                self.elu_alpha = value

    def add_dropouts(self, layer, fraction):
        drop_size = np.ceil(fraction*self.layers[layer]['nodes'])
        node_id = random.sample(
            range(self.layers[layer]['nodes']), int(drop_size))
        for j in node_id:
            self.dropout_nodes[layer][j-1] = True

    def add_regularization(self, name='L1_norm', penalty=0.1):
        self.penalty = penalty
        self.regularization = name

    def add_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer
        for key, value in kwargs.items():
            if (key == 'momentum'):
                self.momentum = value
            elif (key == 'epsilon'):
                self.epsilon = value
            elif (key == 'beta'):
                self.beta = value
            elif (key == 'beta1'):
                self.beta1 = value
            elif (key == 'beta2'):
                self.beta2 = value

    def preceptron(self, inputs, weights, bias, activation_func=None):
        if (activation_func is None):
            activation_func = 'linear'
        output = np.dot(inputs, weights)+bias
        return self.activation_functions[activation_func](output)

    def forward_propagate(self, x, weights=None, bias=None):
        self.activations[0] = x
        for i in range(1, len(self.layers)):
            for j in range(self.layers[i]['nodes']):
                if (self.dropout_nodes[i][j] == True and self.layers[i]['dropouts'] == True):
                    self.activations[i][j] = 0
                else:
                    self.activations[i][j] = self.preceptron(
                        self.activations[i-1], self.weights[i-1][j], self.bias[i][j], self.layers[i]['activation_function'])
        return self.activations[len(self.layers)-1]

    def back_propagate(self, y, p):
        error = float('inf')
        if (self.loss_func == 'mse'):
            error = -(y-p)
        elif (self.loss_func == 'binary_cross_entropy'):
            error = -(y/p)+((1-y)/(1-p))

        for i in reversed(range(len(self.derivatives_w))):
            activation_func = self.activation_functions[self.layers[i+1]['activation_function']]
            delta_w = error * activation_func(self.activations[i+1], derive=True)
            delta_b = error * activation_func(self.activations[i+1], derive=True)
            delta_w_re = delta_w.reshape(delta_w.shape[0], -1)
            activation_re = self.activations[i].reshape(self.activations[i].shape[0], -1)
            self.derivatives_w[i] = np.dot(delta_w_re, activation_re.T)
            self.derivatives_b[i+1] = delta_b
            error = np.dot(self.weights[i].T, delta_w)

    def fit(self, x, y, learning_rate=0.001, epochs=50, batch_size=None, show_loss=False, early_stopping=False,patience=2):
        loss = float('inf')
        patience_count=0
        if (batch_size is None):
            batch_size = x.shape[0]
        for i in range(epochs):
            sum_errors = 0
            if (batch_size != x.shape[0]):
                batch = np.random.choice(x.shape[0], batch_size)
            else:
                batch = np.arange(x.shape[0])
                batch_size = x.shape[0]
            for j in batch:
                output = self.forward_propagate(x[j])
                self.back_propagate(y[j], output)
                self.optimizer_function[self.optimizer](learning_rate)
                sum_errors += self.loss_functions[self.loss_func](y[j], output)

            loss = sum_errors / batch_size
            if (early_stopping == True and len(self.history['Losses']) > 1 and loss > self.history['Losses'][-1]):
                print(
                    "\n<================(EARLY STOPPING AT --> EPOCH {})==================> ".format(i))
                patience_count+=1
                if(patience_count>=patience):
                    break
            self.history['Losses'].append(loss)
            self.history['Weights'].append(self.weights)
            self.history['Bias'].append(self.bias)
            self.history['Activations'].append(self.activations)
            if (show_loss):
                print("Loss: {} =================> at epoch {}".format(loss, i+1))
        print("\nFinal Minimised Loss : {}".format(self.history['Losses'][-1]))
        print("\nTraining complete!")
        print("================================================ :)")
        return self.history['Losses']

    def predict(self, x):
        outputs = []
        for j, val in enumerate(x):
            values = val
            for i in range(1, len(self.layers)):
                z = np.dot(values, self.weights[i-1].T)+self.bias[i]
                values = self.activation_functions[self.layers[i]['activation_function']](
                    z)
            outputs.append(values)

        return np.array(outputs).reshape(-1, self.n_outputs)
