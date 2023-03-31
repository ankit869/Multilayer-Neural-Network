import numpy as np
class convolutional_network():
    def __init__(self):
        self.c_weights = []
        self.c_bias = []
        self.c_activations = []
        self.c_outputs = []
        self.pool_outputs=[]
        self.c_Vdb = []
        self.c_Vdw = []
        self.c_Mdw = []
        self.c_Mdb = []
        self.c_derivatives_w = []
        self.c_derivatives_b = []
    
    def get_pool(self,x,ptype='max'):
        if(ptype=='max'):
            return np.max(x)
        elif(ptype=='min'):
            return np.min(x)
        elif(ptype=='avg'):
            return np.mean(x)
    
    def get_conv_output_shape(self,input_shape,nfilters,kernal_shape,padding='valid',strides=1,pad_size=1,Type='2D'):
        xKernShape,yKernShape=kernal_shape
        xInput,yInput=input_shape

        if Type=='3D':
            xInput,yInput,zInput=input_shape
        else:
            xInput,yInput=input_shape

        if(padding=='valid'):
            pad_size=0

        elif(padding=='same' and  strides==1):
            pad_size=1

        elif(padding=='custom'):
            if(pad_size<1):
                print('ERROR : pad_size must be greater than equal to 1 for type-custom')

        xOutput = int(((xInput - xKernShape + 2 * pad_size) / strides) + 1)
        yOutput = int(((yInput - yKernShape + 2 * pad_size) / strides) + 1)

        if Type=='3d':
            out_size=(zInput,)+(xOutput, yOutput)+(nfilters,)
        else:
            out_size=(None,)+(xOutput, yOutput)+(nfilters,)
        
        return out_size

    def get_pool_output_shape(self,input_shape,nfilters,size=(2,2),padding='valid',strides=1,pad_size=1,Type='2D'):
        if Type=='3D':
            xInput,yInput,zInput=input_shape
        else:
            xInput,yInput=input_shape

        if(padding=='valid'):
            pad_size=0
        elif(padding=='same' and  strides==1):
            pad_size=1
        elif(padding=='custom'):
            if(pad_size<1):
                print('ERROR : pad_size must be greater than equal to 1 for type-custom')

        xOutput = int(((xInput - size[0] + 2 * pad_size) / strides) + 1)
        yOutput = int(((yInput - size[1] + 2 * pad_size) / strides) + 1)
        
        if Type=='3d':
            out_size=(zInput,)+(xOutput, yOutput)+(nfilters,)
        else:
            out_size=(None,)+(xOutput, yOutput)+(nfilters,)
        
        return out_size

    def pool2d(self,Input,pool_type='max',size=(2,2),padding='valid',strides=1,pad_size=1):
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
            return np.mmean(Input)
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

    def conv2d(self,Input,kernal,padding='valid',strides=1,pad_size=1):
        xKernShape=kernal.shape[0]
        yKernShape=kernal.shape[1]
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
                        output[i,j]=np.sum(kernal*(InpPadded[x:x+xKernShape,y:y+yKernShape]))
                        i+=1
                j+=1
        return output

    def conv3d(self,Input,kernal,padding='valid',strides=1,pad_size=1):
        output=[]
        for i in range(Input.shape[2]):
            output.append(self.conv2d(Input[:,:,i],kernal,padding,strides,pad_size).T)
        return np.array(output,np.int32).T
    
    def pool3d(self,Input,pool_type='max',size=(2,2),padding='valid',strides=1,pad_size=1):
        output=[]
        for i in range(Input.shape[2]):
            output.append(self.pool2d(Input[:,:,i],pool_type,size,padding,strides,pad_size).T)
    
        return np.array(output,np.int32).T
    
    def create_layer(self,nfilters=30,filter_size=(3,3),)

    