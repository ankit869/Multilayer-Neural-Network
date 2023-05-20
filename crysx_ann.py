import numpy as np
from opt_einsum import contract, contract_expression

def Softmax(x):
    max_x = np.zeros((x.shape[0],1),dtype=x.dtype)
    for i in range(x.shape[0]):
        max_x[i,0] = np.max(x[i,:])
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=1).reshape((-1, 1)) # Alternative of keepdims=True for Numba compatibility

def Softmax_grad(x): # Best implementation (VERY FAST)
    s = Softmax(x)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=x.dtype)
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=x.dtype)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            for k in range(s.shape[1]):
                temp1[i,j,k] = s[i,j]*a[j,k]
                temp2[i,j,k] = s[i,j]*s[i,k]
    
    return temp1-temp2

def Sigmoid(x): # Also known as logistic/soft step or even expit in scipy.special
    output = np.zeros((x.shape[0],x.shape[1]),dtype=x.dtype)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_val = x[i,j]
            if x_val>=0:
                output[i,j] = 1. / ( 1. + np.exp(-x_val) )
            else:
                e_x = np.exp(x_val)
                output[i,j] = e_x / ( 1. + e_x )
    return output

def Sigmoid_grad(x):
    e_x = np.exp(-x)
    return e_x/(e_x+1)**2

def ReLU(x):
    return x * (x > 0) # This has been the fastest till date

def ReLU_grad(x):
    return np.greater(x, 0.).astype(x.dtype)

def Tanh_offset(x):
    return 0.5*(1.+np.tanh(x))

def Tanh_offset_grad(x):
    return 1./(np.cosh(2.*x)+1.)

def Tanh(x):
    return np.tanh(x)

def Tanh_grad(x):
    return 1.-np.tanh(x)**2 # sech^2{x}

def Identity(x):
    return x

def Identity_grad(x):
    return np.ones(x.shape, dtype=x.dtype)

def Softplus(x): 
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def Softplus_grad(x): # This is simply the sigmoid function
    return Sigmoid(x)
    
def MSE_loss(outi, out0):
    loss = 0.0
    for i in range(outi.shape[0]):
        for j in range(outi.shape[1]):
            loss = loss + (outi[i,j] - out0[i,j])**2 # should have given a race condition but somehow numba is able to avoid it
    loss = loss / outi.shape[1]
    return loss


def MSE_loss_grad(outi, out0): 
    return 2*(outi-out0)/outi.shape[1]

def MAE_loss(predictions, targets):
    loss = 0.0
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            loss = loss + np.abs(predictions[i,j] - targets[i,j]) # should have given a race condition but somehow numba is able to avoid it
    # Average over number of output nodes
    loss = loss / predictions.shape[1]
    return loss


def MAE_loss_grad(predictions, targets):
    loss_grad = np.where(predictions >= targets, 1.0, -1.0)
    return loss_grad/predictions.shape[1]
   
def BCE_loss(predictions, targets, epsilon=1e-7):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    loss = 0.0
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            temp = -(targets[i,j]*np.log(predictions[i,j]+epsilon) + (1.-targets[i,j])*np.log(1.-predictions[i,j]+epsilon))
            loss = loss + temp # should have given a race condition but somehow numba is able to avoid it
    loss = loss / predictions.shape[1]
    return loss

def BCE_loss_grad(predictions, targets):
    return -(np.nan_to_num(np.divide(targets,predictions,dtype=targets.dtype))-np.nan_to_num(np.divide(1.-targets,1.-predictions,dtype=targets.dtype)))/predictions.shape[1]

def CCE_loss(predictions, targets, epsilon=1e-9):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    cce = -np.sum(targets*np.log(predictions+epsilon))
    return cce

def CCE_loss_grad(predictions, targets, epsilon=1e-9):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.nan_to_num(np.divide(targets,predictions,dtype=targets.dtype))


act_func_dict = {'Sigmoid':Sigmoid,'ReLU':ReLU,'Softmax':Softmax, \
    'Tanh':Tanh, 'Tanh_offset':Tanh_offset, 'Identity':Identity, 'Softplus':Softplus}

act_func_grad_dict = {'Sigmoid':Sigmoid_grad,'ReLU':ReLU_grad,'Softmax':Softmax_grad,\
     'Tanh':Tanh_grad, 'Tanh_offset':Tanh_offset_grad, 'Identity':Identity_grad, 'Softplus':Softplus_grad}

loss_func_dict = {'MAE':MAE_loss,'MSE':MSE_loss,'BCE':BCE_loss, \
    'CCE':CCE_loss}

loss_func_grad_dict = {'MAE':MAE_loss_grad,'MSE':MSE_loss_grad,'BCE':BCE_loss_grad, \
    'CCE':CCE_loss_grad}

def init_params(nInputs, neurons_per_layer, method='random2',dtype='float32'):
    nLayers = len(neurons_per_layer)
    weights = [None] * (nLayers)
    biases = [None] * (nLayers)
    for i in range(nLayers):
        if method=='random1':
            if i==0:
                weights[i] = np.random.uniform(low=0.0, high=1.0, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.uniform(low=0.0, high=1.0, size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='random2':
            if i==0:
                weights[i] = np.random.uniform(low=-0.3, high=0.3, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.uniform(low=-0.3, high=0.3, size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        if method=='random3':
            if i==0:
                weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        if method=='Xavier':
            if i==0:
                sqrtN = np.sqrt(nInputs)
                weights[i] = np.random.uniform(low=-1./sqrtN, high=1./sqrtN, size=(neurons_per_layer[i], nInputs))
            else:
                sqrtN = np.sqrt(neurons_per_layer[i-1])
                weights[i] = np.random.uniform(low=-1./sqrtN, high=1./sqrtN, size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='NormXavier':
            if i==0:
                sqrtN = np.sqrt(nInputs)
                sqrtM = np.sqrt(neurons_per_layer[i])
                weights[i] = np.random.uniform(low=-6./(sqrtN+sqrtM), high=6./(sqrtN+sqrtM), size=(neurons_per_layer[i], nInputs))
            else:
                sqrtN = np.sqrt(neurons_per_layer[i-1])
                sqrtM = np.sqrt(neurons_per_layer[i])
                weights[i] = np.random.uniform(low=-6./(sqrtN+sqrtM), high=6./(sqrtN+sqrtM), size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='He':
            if i==0:
                weights[i] = np.random.normal(loc=0.0, scale=np.sqrt(2./nInputs), size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.normal(loc=0.0, scale=np.sqrt(2./neurons_per_layer[i-1]), size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        biases[i] = np.zeros(neurons_per_layer[i])

    if dtype=='float32':
        for i in range(len(weights)):
            weights[i] = weights[i].astype(np.float32)
            biases[i] = biases[i].astype(np.float32)

    return weights, biases



def softmaxTimesVector(a,b): 
    output = contract('ijk,ik->ij',a,b, dtype=a.dtype)
    return output

def forward_feed(x, nLayers, weights, biases, activationFunc):
    a = [None] * (nLayers+1)
    z = [None] * nLayers
    a[0] = x
    for l in range(1,nLayers+1):
        # z[l-1] = np.einsum('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        z[l-1] = contract('ij,kj->ik',a[l-1],weights[l-1],dtype=x.dtype)+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        actFunc_layer = act_func_dict[activationFunc[l-1]] #Activation function for this layer
        a[l] = actFunc_layer(z[l-1]).astype(x.dtype)
    return a, z

def back_propagation_fast(z, a, sigmaPrime, nLayers, nSamples, weights, biases, eeta, dc_daL):

    nSamples = a[0].shape[0]
    delta = [None] * (nLayers+1)
    derWeights = [None] * nLayers
    derBiases = [None] * nLayers
    
    sigmaPrime_layer = act_func_grad_dict[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]),dc_daL).astype(z[0].dtype)
    else:
        delta[nLayers] = (sigmaPrime_layer(z[nLayers-1])*dc_daL).astype(z[0].dtype)
    
    newWeights = weights[:]#.copy()
    newBiases = biases[:]#.copy()
    
    derWeights[nLayers-1] = contract('ji,jk->ik',delta[nLayers],a[nLayers-1])
    newWeights[nLayers-1] = weights[nLayers-1] - eeta*derWeights[nLayers-1]
    derBiases[nLayers-1] = np.sum(delta[nLayers],axis=0)
    newBiases[nLayers-1] = biases[nLayers-1] - eeta*derBiases[nLayers-1]
    for l in range(nLayers-1,0,-1):
        temp = contract('ik,lk->li',weights[l].T, delta[l+1], dtype=z[0].dtype)
        sigmaPrime_layer = act_func_grad_dict[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]),temp).astype(z[0].dtype)
        else:    
            delta[l] = (sigmaPrime_layer(z[l-1])*temp).astype(z[0].dtype)
            
        derWeights[l-1] = contract('ji,jk->ik',delta[l],a[l-1])
        derBiases[l-1] = np.asarray(np.sum(delta[l],axis=0))
        newWeights[l-1] = weights[l-1] - eeta*derWeights[l-1] 
        newBiases[l-1] = biases[l-1] - eeta*derBiases[l-1]
    
    return derWeights, derBiases, newWeights, newBiases


def nn_optimize_fast(inputs, outputs, activationFunc, nLayers, nEpochs=10, batchSize=None, eeta=0.5, weights=None, biases=None, errorFunc=MSE_loss, gradErrorFunc=MSE_loss_grad, get_accuracy=False):
  
    if batchSize==None:
        batchSize = min(32, inputs.shape[0])
    if weights == None:
        weights = []
    if biases == None:
        biases = []
    errors=[]
    nBatches = int(inputs.shape[0]/batchSize)
    if get_accuracy:
        accuracies = []
    for iEpoch in range(nEpochs):
        errorEpoch = 0.0
        if get_accuracy:
            accuracy_epoch = 0.0
        # for iBatch in range(nBatches):
        for iBatch in range(nBatches):
            offset = iBatch*batchSize
            x = inputs[offset:offset + batchSize,:]# Input vector
          
            outExpected = outputs[offset:offset + batchSize,:] # Expected output
            a, z = forward_feed(x, nLayers, weights, biases, activationFunc)

            if get_accuracy:
                bool_mask = np.argmax(outExpected,axis=1)==np.argmax(a[nLayers],axis=1)
                accuracy_epoch += np.sum(bool_mask)/batchSize
            errorBatch = errorFunc(a[nLayers],outExpected)
            errorEpoch += errorBatch/batchSize
            dc_daL = gradErrorFunc(a[nLayers], outExpected)
            dc_daL = dc_daL/batchSize
            derWeights, derBiases, weights, biases = back_propagation_fast(z, a, activationFunc, nLayers, batchSize, weights, biases, eeta, dc_daL)

        errors.append(errorEpoch/nBatches)
        if get_accuracy:
            accuracies.append(accuracy_epoch/nBatches)
        
        if(iEpoch==0):
            print('Average Error with initial weights and biases:', errorEpoch/nBatches)
    

    if get_accuracy:
        return weights, biases, errors, accuracies
    else:
        return weights, biases, errors

class nn_model:
    def __init__(self, nInputs=None, neurons_per_layer=None, activation_func_names=None, batch_size=None,device='CPU', init_method='Xavier'): 
        if nInputs is None:
            print('ERROR: You need to specify the number of input nodes.')
            return
        else: 
            self.nInputs = nInputs
        if neurons_per_layer is None:
            print('ERROR: You need to specify the number of neurons per layer (excluding the input layer) and supply it as a list.')
            return
        else: 
            self.neurons_per_layer = neurons_per_layer
        if activation_func_names is None:
            print('ERROR: You need to specify the activation function for each layer and supply it as a list.')
            return
        else: 
            self.activation_func_names = activation_func_names
        print('Note: The model will use the following device for all the computations: ', device)
        
        self.batch_size = batch_size
        self.device = device
        self.init_method = init_method
        self.init_weights, self.init_biases = init_params(self.nInputs, self.neurons_per_layer, method=self.init_method)
        self.nLayers = len(neurons_per_layer)
        self.weights = self.init_weights
        self.biases = self.init_biases
        self.errors = []
        self.accuracy = []
        self.opt_method = 'SGD'
        self.lr = 0.5

    def init_params(self, method=None):
        if method is None:
            method = self.init_method
        self.init_weights, self.init_biases = init_params(self.nInputs, self.neurons_per_layer, method=method)
        self.weights = self.init_weights
        self.biases = self.init_biases
      
    def details(self):
        print('----------------------------------------------------------------------------------')
        print('****Neural Network Model Details****')
        print('----------------------------------------------------------------------------------')
        print('Number of input nodes: ', self.nInputs)
        print('Number of layers (hidden+output): ', self.nLayers)
        print('Number of nodes in each layer (hidden & output): ', self.neurons_per_layer)
        print('Activation function for each layer (hidden & output):  ', self.activation_func_names)
        print('Method used for weights and biases initialization:  ', self.init_method)
        print('Batch Size: ', self.batch_size)
        print('Device: ', self.device)
        print('Optimization method: ', self.opt_method)
        print('Learning rate: ', self.lr)
        print('----------------------------------------------------------------------------------')
        
    def optimize(self, inputs, outputs, method=None, lr=None, nEpochs=100,loss_func_name=None, miniterEpoch=1,batchProgressBar=False,miniterBatch=100, get_accuracy=False):
        if method is None:
            method = self.opt_method
        if lr is None:
            lr = self.lr

        if loss_func_name is None:
            loss_func = MSE_loss
            loss_func_grad = MSE_loss_grad
        else:
            loss_func = loss_func_dict[loss_func_name]
            loss_func_grad = loss_func_grad_dict[loss_func_name]
        if get_accuracy:
            self.weights, self.biases, self.errors, self.accuracy = nn_optimize_fast(inputs, outputs, self.activation_func_names, self.nLayers, nEpochs=nEpochs, batchSize=self.batch_size, eeta=lr, weights=self.weights, biases=self.biases, errorFunc=loss_func, gradErrorFunc=loss_func_grad, get_accuracy=get_accuracy)
        else:
            self.weights, self.biases, self.errors = nn_optimize_fast(inputs, outputs, self.activation_func_names, self.nLayers, nEpochs=nEpochs, batchSize=self.batch_size, eeta=lr, weights=self.weights, biases=self.biases, errorFunc=loss_func, gradErrorFunc=loss_func_grad)
      
    def save_model_weights(self, filename):
        np.savez(filename, *self.weights)

    
    def save_model_biases(self, filename):
        np.savez(filename, *self.biases)

    def load_model_weights(self, filename):   
        outfile = np.load(filename+'.npz')
        for i in range(len(outfile.files)):
            self.weights[i] = outfile[outfile.files[i]]
    
        
    def load_model_biases(self, filename):
        outfile = np.load(filename+'.npz')
        for i in range(len(outfile.files)):
            self.biases[i] = outfile[outfile.files[i]]
        
    def predict(self, inputs, outputs=None, loss_func_name=None, get_accuracy=False):
        error = 0.0
        accuracy = 0.0
        nBatches = np.maximum(int(inputs.shape[0]/self.batch_size),1)
        
        if inputs.shape[0]/self.batch_size<1:
            nBatches= 1
            batch_size = inputs.shape[0]
        else:
            batch_size = self.batch_size

        predictions = np.zeros([inputs.shape[0], self.neurons_per_layer[-1]],dtype=inputs.dtype)
        if loss_func_name is None:
            loss_func = MSE_loss
            loss_func_grad = MSE_loss_grad
        else:
            loss_func = loss_func_dict[loss_func_name]
            loss_func_grad = loss_func_grad_dict[loss_func_name]
        for iBatch in range(nBatches):
            offset = iBatch*batch_size
            x = inputs[offset:offset + batch_size,:]# Input vector
            
            a, z = forward_feed(x, self.nLayers, self.weights, self.biases, self.activation_func_names)
            new_outputs = a[self.nLayers] 
            predictions[offset:offset + batch_size,:] = new_outputs
            if outputs is not None:
                outExpected = outputs[offset:offset + batch_size,:] # Expected output
                error += loss_func(new_outputs, outExpected)/batch_size
                if get_accuracy:
                    bool_mask = np.argmax(new_outputs,axis=1)==np.argmax(outExpected,axis=1)
                    accuracy += np.sum(bool_mask)
       
        if outputs is None:
            return predictions
        else:
            if not get_accuracy:
                return predictions, error/nBatches
            else :
                return predictions, error/nBatches, accuracy/outputs.shape[0]