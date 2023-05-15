'''
This is the complete code of MultiLayer Neural Network

Developed by: 
    - Ankit Kohli (Student at Delhi University)
    - ankitkohli181@gmail.com (mail)

Have fun!

'''

import numpy as np
import random
import time
from conv_mlp import Convolutional_network
class Activation_Functions():
    def __init__(self):
        super(Activation_Functions, self).__init__()
        self.leaky_relu_fraction = 0.01
        self.elu_alpha = 0.1

    def sigmoid(self, z, derive=False):
        '''
        This Clipping is necessary before calculating exponents
        , because exp(large_number) will produce overflow error

        so clipping inputs with max range of 700
        will help to handle this error.
        
        >>np.exp(700) is almost close to inf but not inf

        >>np.exp([>700]) will produce overflow error

        '''
        z=np.clip(z,a_min=-700,a_max=700)
        if (derive):
            return (z*(1-z))
        else:
            if (isinstance(z,np.ndarray)):
                return np.array([1/(1+np.exp(-i)) for i in z])
            else:
                return 1/(1+np.exp(-z))

    def softmax(self, z, derive=False):
        z=np.clip(z,a_min=-700,a_max=700)
        if (derive):
            if (z.ndim == 2):
                return [self.softmax(i, derive=True) for i in z]
            else:
                '''
                >>Method-1
                  This is the Vectorized method for calculating jacobian matrix
                  Reference-https://medium.com/intuitionmath/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d

                  z=z.reshape(-1,1)
                  return (np.diagflat(z)-np.dot(z,z.T))

                >>Method-2
                  Reference-https://e2eml.school/softmax.html#:~:text=Backpropagation,respect%20to%20its%20output%20values

                  z=z.reshape(1,-1)
                  return (z * np.identity(z.size)-z.transpose() @ z)

                  Note-This method will require gradients of same shape as (1,-1), Otherwise It will produce error.

                '''
                z = z.reshape(-1,1)
                return (np.diagflat(z)-np.dot(z, z.T))

        else:
            '''
            Another method in case of softmax for handling overflow error without clipping

            [np.exp(inputs-max(inputs))]

            This is also done for mathematical convenience
            and for removing overflow error if occured
            (outputs will have no effect)

            '''
            exp_vals = np.exp(z)
            return exp_vals/np.sum(exp_vals+self.epsilon)

    def relu(self, z, derive=False):
        if (derive):
            return np.where(z > 0, 1, 0)
        else:
            if (isinstance(z,np.ndarray)):
                return np.array([max(0, i) for i in z])
            else:
                return max(0, z)

    def tanh(self, z, derive=False):
        z=np.clip(z,a_min=-700,a_max=700)
        if (derive):
            if (isinstance(z,np.ndarray)):
                return np.array([(1-i**2) for i in z])
            else:
                return 1-z**2
        else:
            if (isinstance(z,np.ndarray)):
                return np.array([((np.exp(i)-np.exp(-i))/(np.exp(i)+np.exp(-i)+self.epsilon)) for i in z])
            else:
                return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)+self.epsilon)

    def elu(self, z, derive=False):
        z=np.clip(z,a_min=-700,a_max=700)
        if (derive):
            return np.where(z > 0, 1, z+self.elu_alpha)
        else:
            return np.where(z > 0, z, self.elu_alpha*(np.exp(z)+self.elu_alpha))

    def leaky_relu(self, z, derive=False):
        if (derive):
            return np.where(z >= 0, 1, self.leaky_relu_fraction)
        else:
            if (isinstance(z,np.ndarray)):
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
        super(Losses, self).__init__()

    def mse(self, y, p):
        return np.mean((y - p) ** 2)

    def binary_cross_entropy(self, y, p):
        return np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))

    def categorical_cross_entropy(self, y, p):
        return np.mean(-np.sum(y*np.log(p)))

class instance_variables:

    def __init__(self):
        super(instance_variables, self).__init__()
        self.weights = []
        self.bias = []
        self.Vdb = []
        self.Vdw = []
        self.Mdw = []
        self.Mdb = []
        self.derivatives_w = []
        self.derivatives_b = []
        self.regularization = None
        self.activations = []
        self.dropout_nodes = []
        self.layers = []
        self.convolution=False

class Weight_Initalizer(instance_variables):
    def __init__(self):
        super(Weight_Initalizer, self).__init__()
        self.weight_initializer = {
            'random_uniform': self.random_uniform,
            'random_normal': self.random_normal,
            'glorot_uniform': self.glorot_uniform,
            'glorot_normal': self.glorot_normal,
            'he_uniform': self.he_uniform,
            'he_normal': self.he_normal
        }

    def random_uniform(self, seed=None, args=dict()):
        minval = -0.05
        maxval = 0.05
        for key, value in args.items():
            if (key == 'minval'):
                minval = value
            elif (key == 'maxval'):
                maxval = value
            elif (key == 'seed'):
                np.random.seed(seed)

        for i in range(len(self.layers)-1):
            self.weights[i] = np.random.uniform(minval, maxval, size=(
                self.layers[i+1]['nodes'], self.layers[i]['nodes']))
    
    def random_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            self.weights[i] = np.random.randn(self.layers[i+1]['nodes'], self.layers[i]['nodes'])

    def glorot_uniform(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            limit = np.sqrt(6 / (self.layers[i+1]['nodes'] + self.layers[i]['nodes']))
            vals = np.random.uniform(-limit, limit,size=(self.layers[i+1]['nodes'], self.layers[i]['nodes']))
            self.weights[i] = vals

    def glorot_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            limit = np.sqrt(
                2 / (self.layers[i+1]['nodes'] + self.layers[i]['nodes']))
            vals = np.random.randn(self.layers[i+1]['nodes'], self.layers[i]['nodes'])*limit
            self.weights[i] = vals

    def he_uniform(self, seed=None, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            limit = np.sqrt(6 / (self.layers[i]['nodes']))
            vals = np.random.uniform(-limit, limit,size=(self.layers[i+1]['nodes'], self.layers[i]['nodes']))
            self.weights[i] = vals

    def he_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(len(self.layers)-1):
            vals = np.random.randn(
                self.layers[i+1]['nodes'], self.layers[i]['nodes']) * np.sqrt(2/(self.layers[i]['nodes']))
            self.weights[i] = vals


class Optimizers(Weight_Initalizer):
    def __init__(self):
        super(Optimizers, self).__init__()
        self.epsilon = 1e-07
        self.momentum = 0.9
        self.beta = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.optimizer = 'RMSprop'
        self.optimizer_function = {
            'momentum': self.Momentum,
            'gradient_descent': self.gradient_descent,
            'AdaGrad': self.AdaGrad,
            'RMSprop': self.RMSprop,
            'Adam': self.Adam
        }

    def get_gradients(self, layer):
        derivatives_w = None
        derivatives_b = None
        if (self.regularization == 'L2_norm'):
            derivatives_w = self.derivatives_w[layer] + (2*self.penalty*self.weights[layer])
        elif (self.regularization == 'L1_norm'):
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
            self.weights[i] -= learningRate*( dw/np.sqrt(self.Vdw[i]+self.epsilon))
            self.bias[i+1] -= learningRate*(db/np.sqrt(self.Vdb[i+1]+self.epsilon))

    def RMSprop(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Vdw[i] = self.beta*self.Vdw[i]+(1-self.beta)*(dw**2)
            self.Vdb[i+1] = self.beta*self.Vdb[i+1]+(1-self.beta)*(db**2)
            self.weights[i] -= learningRate*( dw/np.sqrt(self.Vdw[i]+self.epsilon))
            self.bias[i+1] -= learningRate*(db/np.sqrt(self.Vdb[i+1]+self.epsilon))

    def Adam(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Mdw[i] = self.beta1*self.Mdw[i]+(1-self.beta1)*dw
            self.Vdw[i] = self.beta2*self.Vdw[i]+(1-self.beta2)*(dw**2)
            m_dw = self.Mdw[i]/(1-self.beta1)
            v_dw = self.Vdw[i]/(1-self.beta2)
            self.weights[i] -= learningRate*(m_dw/np.sqrt(v_dw+self.epsilon))

            self.Mdb[i+1] = self.beta1*self.Mdb[i+1]+(1-self.beta1)*db
            self.Vdb[i+1] = self.beta2*self.Vdb[i+1]+(1-self.beta2)*(db**2)
            m_db = self.Mdb[i+1]/(1-self.beta1)
            v_db = self.Vdb[i+1]/(1-self.beta2)
            self.bias[i+1] -= learningRate*(m_db/np.sqrt(v_db+self.epsilon))

class ANN_propagate(Activation_Functions, Losses, Optimizers):
    def __init__(self):
        super(ANN_propagate, self).__init__()
    
    def forward_propagate(self, x):
        self.activations[0] = x
        for i in range(1, len(self.layers)):
            z = np.dot(self.activations[i-1], self.weights[i-1].T)+self.bias[i]
            self.activations[i] = self.activation_functions[self.layers[i]['activation_function']](z)

            if (self.layers[i]['dropouts'] == True):
                self.activations[i][np.where(self.dropout_nodes[i])] = 0

        return self.activations[len(self.layers)-1]

    def back_propagate(self, y, p):
        error = float('inf')
        if (self.loss_func == 'mse'):
            error = -2*(y-p)
        elif (self.loss_func == 'binary_cross_entropy'):
            p += self.epsilon
            error = -(y/p)+((1-y)/(1-p))
        elif (self.loss_func == 'categorical_cross_entropy'):
            p += self.epsilon
            error = -(y/p)

        for i in reversed(range(len(self.derivatives_w))):
            delta_w = None
            delta_w_re = None
            func_name = self.layers[i+1]['activation_function']
            activation_func = self.activation_functions[func_name]

            if (func_name == 'softmax'):
                delta_w = np.dot(error,activation_func(self.activations[i+1], derive=True))
            else:
                delta_w = error * activation_func(self.activations[i+1], derive=True)

            delta_b = delta_w
            delta_w_re = delta_w.reshape(delta_w.shape[0], -1)
            activation_re = self.activations[i].reshape(self.activations[i].shape[0], -1)
            self.derivatives_w[i] = np.dot(delta_w_re,activation_re.T)
            self.derivatives_b[i+1] = delta_b
            error = np.dot(self.weights[i].T,delta_w)

class MultiLayerNeuralNetwork(ANN_propagate,Convolutional_network):
    def __init__(self):
        super(MultiLayerNeuralNetwork, self).__init__()
        self.history = {'Losses': [], 'Weights': [],'Bias': [], 'Activations': [], 'Conv_Weights': [], 'Conv_Bias': [], 'Conv_Activations': []}
        self.loss_functions = {
            "mse": self.mse,
            "binary_cross_entropy": self.binary_cross_entropy,
            "categorical_cross_entropy": self.categorical_cross_entropy
        }
        
        self.activation_functions = {
            "sigmoid": self.sigmoid,
            "softmax": self.softmax,
            "relu": self.relu,
            "leaky_relu": self.leaky_relu,
            "elu": self.elu,
            "tanh": self.tanh,
            "linear": self.linear
        }

    def show_summary(self):
        print(f'''
        {'( MODEL SUMMARY )'.center(80)}
        
        ==================================================================================
        {"Layer".center(20)}{"Activation Function".center(20)}{"Output Shape".center(20)}{"Params".center(20)}
        ==================================================================================''')
        p_sum=0
        if self.convolution is True:
            for i in range(len(self.c_layers)):
                p_sum+=self.c_layers[i]['params']
                print(f'''
        {self.c_layers[i]['type'].center(20)}{str(self.c_layers[i]['activation_function']).center(20)}{str(self.c_layers[i]['output_shape']).center(20)}{str(int(self.c_layers[i]['params'])).center(20)}
        ----------------------------------------------------------------------------------''')    
        for i in range(len(self.layers)):
            p_sum+=self.layers[i]['params']
            print(f'''
        {self.layers[i]['type'].center(20)}{str(self.layers[i]['activation_function']).center(20)}{str(self.layers[i]['output_shape']).center(20)}{str(int(self.layers[i]['params'])).center(20)}
        ----------------------------------------------------------------------------------''')
        
        print(f'''
        ==================================================================================

        Total Params (trainable) - {int(p_sum)}
        __________________________________________________________________________________
        ''')
        
    def add_conv2d_layer(self,input_shape=None,nfilters=10,kernel_size=(3,3),activation_function="relu",padding='valid',strides=1,pad_size=1):
        self.convolution=True
        if len(self.c_layers)>=1:
            prev_layer=self.c_layers[-1]
            input_shape=prev_layer['output_shape']

            output_shape=self.get_conv_output_shape(input_shape,nfilters,kernel_size,padding,strides,pad_size)
            self.c_layers.append({'input_shape':input_shape,'nfilters':nfilters, 'kernel_size': kernel_size,'output_shape':output_shape, 'activation_function':activation_function,'padding':padding,'strides':strides,'pad_size':pad_size,'type':'Conv_2D','params':0})
        else:
            output_shape=self.get_conv_output_shape(input_shape,nfilters,kernel_size,padding,strides,pad_size)
            self.c_layers.append({'input_shape':input_shape,'nfilters':nfilters, 'kernel_size': kernel_size,'output_shape':output_shape, 'activation_function':activation_function,'padding':padding,'strides':strides,'pad_size':pad_size,'type':'Conv_2D (Input)','params':0})
 
    def add_pool2d_layer(self,pool_type='max',size=(2,2),padding='valid',strides=1,pad_size=1):
        if len(self.c_layers)>=1:
            prev_layer=self.c_layers[-1]
            if prev_layer['type'].find('Conv')!=-1:
                input_shape=prev_layer['output_shape']                
                output_shape=self.get_pool_output_shape(input_shape,size,padding,strides,pad_size)
                self.c_layers.append({'input_shape':input_shape,'pool_type':pool_type,'size':size,'output_shape':output_shape,'activation_function':'linear','padding':padding,'strides':strides,'pad_size':pad_size,'type':'Pool_2D','params':0})
            else:
                print("ERROR : Adding pooling layer before convolution !!")

   
    def add_layer(self, nodes=3, activation_function='linear',conv_flatten_input=False, input_layer=False, output_layer=False, dropouts=False, dropout_fraction=None, **kwargs):
        if (input_layer is True or conv_flatten_input is True):
            if conv_flatten_input is True:
                out_size=1
                for i in self.c_layers[-1]['output_shape']:
                    if i is not None:
                        out_size*=i
                        
                self.n_inputs = out_size
                nodes=out_size
                self.layers.append({'nodes': nodes, 'activation_function': 'linear', 'dropouts': False,'type':'Conv_flatten','output_shape':(None,nodes),'params':0})
            else:
                self.n_inputs = nodes
                self.layers.append({'nodes': nodes, 'activation_function': 'linear', 'dropouts': False,'type':'Input','output_shape':(None,nodes),'params':0})
        elif (output_layer is not False):
            self.n_outputs = nodes
            self.layers.append({'nodes': nodes, 'activation_function': activation_function, 'dropouts': False,'type':'Output','output_shape':(None,nodes),'params':0})
        else:
            self.layers.append({'nodes': nodes, 'activation_function': activation_function,'dropouts': dropouts, 'dropout_fraction': dropout_fraction,'type':'Dense','output_shape':(None,nodes),'params':0})


    def compile_model(self, loss_function='mse', weight_initializer='glorot_uniform',show_summary=True, **kwargs):
        self.loss_func = loss_function
        for i in range(len(self.layers)):
            self.activations.append(np.zeros(self.layers[i]['nodes']))
            self.bias.append(np.zeros(self.layers[i]['nodes']))
            self.Vdb.append(np.zeros(self.layers[i]['nodes']))
            self.Mdb.append(np.zeros(self.layers[i]['nodes']))
            self.dropout_nodes.append(np.zeros(self.layers[i]['nodes'], dtype=bool))
            self.derivatives_b.append(np.zeros(self.layers[i]['nodes']))

            if (self.layers[i]['dropouts'] == True):
                self.add_dropouts(i, self.layers[i]['dropout_fraction'])

            if i>0:
                self.layers[i]['params']+=self.layers[i]['nodes']

        for i in range(len(self.layers)-1):
            self.Vdw.append(np.zeros((self.layers[i+1]['nodes'], self.layers[i]['nodes'])))
            self.Mdw.append(np.zeros((self.layers[i+1]['nodes'], self.layers[i]['nodes'])))
            self.weights.append(np.random.rand(self.layers[i+1]['nodes'], self.layers[i]['nodes']))
            self.derivatives_w.append(np.zeros((self.layers[i+1]['nodes'], self.layers[i]['nodes'])))

            self.layers[i+1]['params']+=(self.layers[i+1]['nodes']*self.layers[i]['nodes'])

        self.weight_initializer[weight_initializer](kwargs)

        if self.convolution is True:
            self.init_conv_vars()
            self.conv_weight_initializer[weight_initializer](kwargs)

        for key, value in kwargs.items():
            if (key == 'leaky_relu_fraction'):
                self.leaky_relu_fraction = value
            elif (key == 'elu_alpha'):
                self.elu_alpha = value
            elif (key=='use_conv_bias'):
                self.use_conv_bias = value

        if(show_summary):
            self.show_summary()

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

    def check_encoding(self, X):
        return ((X.sum(axis=1)-np.ones(X.shape[0])).sum() == 0)
    
    def fit(self, x, y, learning_rate=0.001, epochs=50, batch_size=1, show_loss=False, early_stopping=False, patience=2):
        loss = float('inf')
        total_time=0
        patience_count = 0
        for i in range(epochs):
            t = Timer()
            t.start()
            sum_errors = 0

            shuffled_indices = np.random.permutation(x.shape[0])
            X_shuffled = x[shuffled_indices]
            Y_shuffled = y[shuffled_indices]

            n_batches = int(np.ceil(x.shape[0] / batch_size))
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                for xi,yi in zip(X_batch,Y_batch):
                    if self.convolution is True:
                        output = self.conv_forward_propagate(xi)
                        # self.conv_back_propagate(y[j], output)
                        # self.conv_optimizer_function[self.optimizer](learning_rate)
                        # output = self.forward_propagate(self.conv_output.flatten())
                        # self.back_propagate(y[j], output)
                        # self.optimizer_function[self.optimizer](learning_rate)
                        print(output.shape)
                        return
                    else:
                        output = self.forward_propagate(xi)
                        self.back_propagate(yi, output)
                        self.optimizer_function[self.optimizer](learning_rate)
                
                sum_errors += self.loss_functions[self.loss_func](yi, output)

            elapse_time=t.stop()
            total_time+=elapse_time
            loss = sum_errors / n_batches

            if (early_stopping == True and len(self.history['Losses']) > 1 and loss > self.history['Losses'][-1]):
                patience_count += 1
                if (patience_count >= patience):
                    print(
                        "\n<==================(EARLY STOPPING AT --> EPOCH {})====================> ".format(i))
                    break

            self.history['Losses'].append(loss)
            self.history['Weights'].append(self.weights)
            self.history['Bias'].append(self.bias)
            self.history['Activations'].append(self.activations)
            if (show_loss):
                print(f"Loss: {loss:.8f} =================> at epoch {i+1}, elapse-time : {elapse_time:.8f} seconds")
        print("\nFinal Minimised Loss : {}".format(self.history['Losses'][-1]))
        print(f"\nTraining complete!! , Average Elapse-Time (per epoch) : {(total_time/epochs):.8f} seconds")
        print("========================================================================= :)")
        return self.history['Losses']

    def predict(self, x):
        outputs = []
        for j, val in enumerate(x):
            values = val
            for i in range(1, len(self.layers)):
                if (self.layers[i-1]['dropouts'] == True):
                    wgt = self.weights[i-1] * self.layers[i-1]['dropout_fraction']
                    z = np.dot(values, wgt.T)+self.bias[i]
                else:
                    z = np.dot(values, self.weights[i-1].T)+self.bias[i]
                values = self.activation_functions[self.layers[i]['activation_function']](z)
            outputs.append(values)

        return np.array(outputs).reshape(-1, self.n_outputs)


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self,show_elapsed=False):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if show_elapsed:
            print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return elapsed_time
