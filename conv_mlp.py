import numpy as np
from neural_cpp_link import py_conv2d_multi_channel,py_conv2d_single_channel,py_pool2d

class conv_instance_variables:
    def __init__(self):
        super(conv_instance_variables, self).__init__()
        self.c_weights = []
        self.c_bias = []
        self.c_activations = []
        self.pool_outputs=[]
        self.conv_output=None
        self.c_Vdb = []
        self.c_Vdw = []
        self.c_Mdw = []
        self.c_Mdb = []
        self.c_derivatives_w = []
        self.c_derivatives_b = []
        self.c_layers=[]

class Conv_Weight_Initalizer(conv_instance_variables):
    def __init__(self):
        super(Conv_Weight_Initalizer, self).__init__()
        self.conv_weight_initializer = {
            'random_uniform': self.conv_random_uniform,
            'random_normal': self.conv_random_normal,
            'glorot_uniform': self.conv_glorot_uniform,
            'glorot_normal': self.conv_glorot_normal,
            'he_uniform': self.conv_he_uniform,
            'he_normal': self.conv_he_normal
        }

    def conv_random_uniform(self, seed=None, args=dict()):
        minval = -0.05
        maxval = 0.05
        for key, value in args.items():
            if (key == 'minval'):
                minval = value
            elif (key == 'maxval'):
                maxval = value
            elif (key == 'seed'):
                np.random.seed(seed)
        j=0
        for i in range(len(self.c_layers)):
            if self.c_layers[i]["type"].find('Conv')!=-1:
                self.c_weights[j] = np.random.uniform(minval, maxval, size=(self.c_weights[j].shape))
                j+=1

    def conv_random_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        j=0
        for i in range(len(self.c_layers)):
            if self.c_layers[i]["type"].find('Conv')!=-1:
                self.c_weights[j] = np.random.randn(self.c_weights[j].shape)
                j+=1

    def conv_glorot_uniform(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
            
        j=0
        for i in range(len(self.c_layers)):
            if self.c_layers[i]["type"].find('Conv')!=-1:
                receptive_field=self.c_layers[i]['kernel_size'][0]*self.c_layers[i]['kernel_size'][1]
                fan_in=receptive_field*self.c_layers[i]['input_shape'][-1]
                fan_out=receptive_field*self.c_layers[i]['output_shape'][-1]
                limit = np.sqrt(6 / (fan_in + fan_out))
                vals = np.random.uniform(-limit, limit,size=(self.c_weights[j].shape))
                self.c_weights[j] = vals
                j+=1

    def conv_glorot_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        j=0
        for i in range(len(self.c_layers)):
            if self.c_layers[i]["type"].find('Conv')!=-1:
                receptive_field=self.c_layers[i]['kernel_size'][0]*self.c_layers[i]['kernel_size'][1]
                fan_in=receptive_field*self.c_layers[i]['input_shape'][-1]
                fan_out=receptive_field*self.c_layers[i]['output_shape'][-1]
                limit = np.sqrt(2/(fan_in + fan_out))
                self.c_weights[j] = np.random.randn(self.c_weights[j].shape)*limit
                j+=1

    def conv_he_uniform(self, seed=None, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        
        j=0
        for i in range(len(self.c_layers)):
            if self.c_layers[i]["type"].find('Conv')!=-1:
                receptive_field=self.c_layers[i]['kernel_size'][0]*self.c_layers[i]['kernel_size'][1]
                fan_in=receptive_field*self.c_layers[i]['input_shape'][-1]
                limit = np.sqrt(6 / (fan_in))
                vals = np.random.uniform(-limit, limit,size=(self.c_weights[j].shape))
                self.c_weights[j] = vals
                j+=1

    def conv_he_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        j=0
        for i in range(len(self.c_layers)):
            if self.c_layers[i]["type"].find('Conv')!=-1:
                receptive_field=self.c_layers[i]['kernel_size'][0]*self.c_layers[i]['kernel_size'][1]
                fan_in=receptive_field*self.c_layers[i]['input_shape'][-1]
                vals = np.random.randn(self.c_weights[i].shape) * np.sqrt(2/(fan_in))
                self.c_weights[j] = vals
                j+=1

class Conv_Optimizers(Conv_Weight_Initalizer):
    def __init__(self):
        super(Conv_Optimizers, self).__init__()
        self.conv_optimizer_function = {
            'momentum': self.conv_Momentum,
            'gradient_descent': self.conv_gradient_descent,
            'AdaGrad': self.conv_AdaGrad,
            'RMSprop': self.conv_RMSprop,
            'Adam': self.conv_Adam
        }

    def conv_get_gradients(self, layer):
        derivatives_w = None
        derivatives_b = None
        if (self.regularization == 'L2_norm'):
            derivatives_w = self.c_derivatives_w[layer] + (2*self.penalty*self.c_weights[layer])
        elif (self.regularization == 'L1_norm'):
            derivatives_w = self.c_derivatives_w[layer]+self.penalty
        else:
            derivatives_w = self.c_derivatives_w[layer]

        derivatives_b = self.derivatives_b[layer+1].flatten()
        return derivatives_w, derivatives_b

    def conv_gradient_descent(self, learningRate=0.001):
        for i in range(len(self.c_layers)-1):
            dw, db = self.conv_get_gradients(i)
            self.c_weights[i] -= dw * learningRate
            self.c_bias[i+1] -= db * learningRate

    def conv_Momentum(self, learningRate=0.001):
        for i in range(len(self.c_layers)-1):
            dw, db = self.conv_get_gradients(i)
            self.c_Vdw[i] = (self.momentum*self.c_Vdw[i]) + (dw*learningRate)
            self.c_Vdb[i+1] = (self.momentum*self.c_Vdb[i+1]) + (db*learningRate)
            self.c_weights[i] -= self.c_Vdw[i]
            self.c_bias[i+1] -= self.c_Vdb[i+1]

    def conv_AdaGrad(self, learningRate=0.001):
        for i in range(len(self.c_layers)-1):
            dw, db = self.conv_get_gradients(i)
            self.c_Vdw[i] = self.c_Vdw[i]+(dw**2)
            self.c_Vdb[i+1] = self.c_Vdb[i+1]+(db**2)
            self.c_weights[i] -= learningRate*( dw/np.sqrt(self.c_Vdw[i]+self.epsilon))
            self.c_bias[i+1] -= learningRate*(db/np.sqrt(self.c_Vdb[i+1]+self.epsilon))

    def conv_RMSprop(self, learningRate=0.001):
        for i in range(len(self.c_layers)-1):
            dw, db = self.conv_get_gradients(i)
            self.c_Vdw[i] = self.beta*self.c_Vdw[i]+(1-self.beta)*(dw**2)
            self.c_Vdb[i+1] = self.beta*self.c_Vdb[i+1]+(1-self.beta)*(db**2)
            self.c_weights[i] -= learningRate*( dw/np.sqrt(self.c_Vdw[i]+self.epsilon))
            self.c_bias[i+1] -= learningRate*(db/np.sqrt(self.c_Vdb[i+1]+self.epsilon))

    def conv_Adam(self, learningRate=0.001):
        for i in range(len(self.c_layers)-1):
            dw, db = self.conv_get_gradients(i)
            self.c_Mdw[i] = self.beta1*self.c_Mdw[i]+(1-self.beta1)*dw
            self.c_Vdw[i] = self.beta2*self.c_Vdw[i]+(1-self.beta2)*(dw**2)
            m_dw = self.c_Mdw[i]/(1-self.beta1)
            v_dw = self.c_Vdw[i]/(1-self.beta2)
            self.c_weights[i] -= learningRate*(m_dw/np.sqrt(v_dw+self.epsilon))

            self.c_Mdb[i+1] = self.beta1*self.c_Mdb[i+1]+(1-self.beta1)*db
            self.c_Vdb[i+1] = self.beta2*self.c_Vdb[i+1]+(1-self.beta2)*(db**2)
            m_db = self.c_Mdb[i+1]/(1-self.beta1)
            v_db = self.c_Vdb[i+1]/(1-self.beta2)
            self.c_bias[i+1] -= learningRate*(m_db/np.sqrt(v_db+self.epsilon))


class Convolutional_network(Conv_Weight_Initalizer):
    def __init__(self):
        super(Convolutional_network, self).__init__()
    
    def get_pool(self,x,ptype='max'):
        if(ptype=='max'):
            return np.max(x)
        elif(ptype=='min'):
            return np.min(x)
        elif(ptype=='avg'):
            return np.mean(x)
    
    def get_conv_output_shape(self,input_shape,nfilters,kernel_shape,padding='valid',strides=1,pad_size=1):
        xKernShape,yKernShape=kernel_shape
        xInput=None
        yInput=None
        if len(input_shape)>3:
            xInput,yInput=input_shape[1:3]
        else:
            xInput,yInput=input_shape[0:2]

        if(padding=='valid'):
            pad_size=0

        elif(padding=='same' and  strides==1):
            pad_size=1

        elif(padding=='custom'):
            if(pad_size<1):
                print('ERROR : pad_size must be greater than equal to 1 for type-custom')

        xOutput = int(((xInput - xKernShape + 2 * pad_size) / strides) + 1)
        yOutput = int(((yInput - yKernShape + 2 * pad_size) / strides) + 1)

        out_size=(None,)+(xOutput, yOutput)+(nfilters,)
        
        return out_size

    def get_pool_output_shape(self,input_shape,size=(2,2),padding='valid',strides=1,pad_size=1):
        xInput=None
        yInput=None

        xInput,yInput,nfilters=input_shape[1:]

        if(padding=='valid'):
            pad_size=0
        elif(padding=='same' and  strides==1):
            pad_size=1
        elif(padding=='custom'):
            if(pad_size<1):
                print('ERROR : pad_size must be greater than equal to 1 for type-custom')

        xOutput = int(((xInput - size[0] + 2 * pad_size) / strides) + 1)
        yOutput = int(((yInput - size[1] + 2 * pad_size) / strides) + 1)
        out_size=(None,)+(xOutput, yOutput)+(nfilters,)
        
        return out_size

    def get_pool(self,x,ptype='max'):
        if(ptype=='max'):
            return np.max(x)
        elif(ptype=='min'):
            return np.min(x)
        elif(ptype=='avg'):
            return np.mean(x)
    
    def pool2d(self,Input,pool_type='max',size=(2,2),padding='valid',strides=1,pad_size=1,cmode=True):

        if(cmode):
            return py_pool2d(Input,pool_type,size,padding,strides,pad_size)

        xInput=Input.shape[0]
        yInput=Input.shape[1]

        output=None
        InpPadded=None
        if(padding=='valid'):
            pad_size=0
            InpPadded=Input
        elif(padding=='same' and  strides==1):
            pad_size=1
            InpPadded=np.zeros((xInput+2*pad_size,yInput+2*pad_size))
            InpPadded[pad_size:-pad_size,pad_size:-pad_size]=Input
        elif(padding=='custom'):
            if(pad_size<1):
                print('ERROR : pad_size must be greater than equal to 1 for type-custom')

            pad_size=pad_size
            InpPadded=np.zeros((xInput+2*pad_size,yInput+2*pad_size))
            InpPadded[pad_size:-pad_size,pad_size:-pad_size]=Input
        
        xOutput = int(((xInput - size[0] + 2 * pad_size) / strides) + 1)
        yOutput = int(((yInput - size[1] + 2 * pad_size) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        if(pool_type=='global_max'):
            return np.max(Input)
        elif(pool_type=='global_min'):
            return np.min(Input)
        elif(pool_type=='global_avg'):
            return np.mean(Input)
        else:
            j=0
            for y in range(InpPadded.shape[1]):
                if y > InpPadded.shape[1] - size[1]:
                    break

                if y%strides==0:
                    i=0
                    for x in range(InpPadded.shape[0]):
                        if x > InpPadded.shape[0] - size[0]:
                            break
                        
                        if x%strides==0:
                            output[i,j]=self.get_pool((InpPadded[x:x+size[0],y:y+size[1]]),ptype=pool_type)
                            i+=1
                    j+=1
            return output
    
    def conv2d(self,Input,kernel,padding='valid',strides=1,pad_size=1,cmode=True):

        multichannel=False
        if(Input.ndim==2 or kernel.ndim==2):
            xKernShape,yKernShape=kernel.shape
            xInput,yInput=Input.shape
            nChannels=1
            nfilters=1
            if(cmode):
                return py_conv2d_single_channel(Input,kernel,padding,strides,pad_size)
        else:
            xKernShape,yKernShape,nfilters=kernel.shape
            xInput,yInput,nChannels=Input.shape
            multichannel=True
            if(cmode):
                return py_conv2d_multi_channel(Input,kernel,padding,strides,pad_size)

        output=None
        InpPadded=None
        if(padding=='valid'):
            pad_size=0
            InpPadded=Input
        elif(padding=='same' and  strides==1):
            pad_size=1
            if(multichannel):
                InpPadded=np.zeros((xInput+2*pad_size,yInput+2*pad_size,nChannels))
                InpPadded[pad_size:-pad_size,pad_size:-pad_size,:]=Input
            else:
                InpPadded=np.zeros((xInput+2*pad_size,yInput+2*pad_size))
                InpPadded[pad_size:-pad_size,pad_size:-pad_size]=Input

        elif(padding=='custom'):
            if(pad_size<1):
                print('ERROR : pad_size must be greater than equal to 1 for type-custom')
            pad_size=pad_size
            if(multichannel):
                InpPadded=np.zeros((xInput+2*pad_size,yInput+2*pad_size,nChannels))
                InpPadded[pad_size:-pad_size,pad_size:-pad_size,:]=Input
            else:
                InpPadded=np.zeros((xInput+2*pad_size,yInput+2*pad_size))
                InpPadded[pad_size:-pad_size,pad_size:-pad_size]=Input

        xOutput = int(((xInput - xKernShape + 2 * pad_size) / strides) + 1)
        yOutput = int(((yInput - yKernShape + 2 * pad_size) / strides) + 1)

        output = np.zeros((xOutput, yOutput))
        j=0

        for y in range(InpPadded.shape[1]):
            if y > InpPadded.shape[1] - yKernShape:
                break
            if y%strides==0:
                i=0
                for x in range(InpPadded.shape[0]):
                    if x > InpPadded.shape[0] - xKernShape:
                        break
                    if x%strides==0:
                        if multichannel is True:
                            for k in range(nfilters):
                                patch=InpPadded[x:x+xKernShape,y:y+yKernShape,k]
                                output[i,j]+=np.sum(kernel[:,:,k]*patch)
                            i+=1
                        else:
                            patch=InpPadded[x:x+xKernShape,y:y+yKernShape]
                            output[i,j]+=np.sum(kernel*patch)
                            i+=1 
                j+=1
        return output

    def init_conv_vars(self):
        for i in range(len(self.c_layers)):
            if self.c_layers[i]['type'].find('Conv')!=-1:
                self.c_activations.append(np.zeros(self.c_layers[i]['output_shape'][1:]))
                wgt_dim=None
                if len(self.c_layers[i]['input_shape'])==2:
                    wgt_dim=self.c_layers[i]['kernel_size']+(self.c_layers[i]['nfilters'],)
                else:
                    wgt_dim=self.c_layers[i]['kernel_size']+(self.c_layers[i]['input_shape'][-1],self.c_layers[i]['nfilters'])

                self.c_weights.append(np.zeros(wgt_dim))
                self.c_Vdw.append(np.zeros(wgt_dim))
                self.c_Mdw.append(np.zeros(wgt_dim))
                self.c_derivatives_w.append(np.zeros(wgt_dim))
            
                w_params=1
                for j in wgt_dim:
                    if j is not None:
                        w_params*=j

                b_dim=(self.c_layers[i]['output_shape'][-1])
                self.c_bias.append(np.zeros(b_dim))
                self.c_Vdb.append(np.zeros(b_dim))
                self.c_Mdb.append(np.zeros(b_dim))
                self.c_derivatives_b.append(np.zeros(b_dim))

                b_params=b_dim

                self.c_layers[i]['params']+=(w_params+b_params)
            else:
                self.pool_outputs.append(np.zeros(self.c_layers[i]['output_shape'][1:]))
    
    def conv_forward_propagate(self,x,cmode=True):
        i=0
        j=0
        for layer in (self.c_layers):
            if layer['type'].find('Conv_2D')!=-1:
                for k in range(layer['nfilters']):
                    kernel=None
                    if len(self.c_layers[i]['input_shape'])==2:
                        kernel=self.c_weights[i][:,:,k]
                    else:
                        kernel=self.c_weights[i][:,:,:,k]

                    if i==0:
                        conv_out=self.conv2d(x,kernel,padding=layer['padding'],strides=layer['strides'],pad_size=layer['pad_size'],cmode=cmode)
                    else:
                        conv_out=self.conv2d(self.conv_output,kernel,padding=layer['padding'],strides=layer['strides'],pad_size=layer['pad_size'],cmode=cmode)
                    
                    output=self.activation_functions[layer['activation_function']](conv_out.flatten())
                    self.c_activations[i][:,:,k]=output.reshape(conv_out.shape)
                self.conv_output=self.c_activations[i]
                i+=1

            else:
                for k in range(len(self.c_activations[i-1])):
                    self.pool_outputs[j][:,:,k]=self.pool2d(self.c_activations[i-1][:,:,k],pool_type=layer['pool_type'],size=layer['size'],padding=layer['padding'],strides=layer['strides'],pad_size=layer['pad_size'],cmode=cmode)
                self.conv_output=self.pool_outputs[j]
                j+=1
        return self.conv_output