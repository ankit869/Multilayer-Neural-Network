# import numpy as np
# from neural_cpp_link import *

# img=np.random.rand(1000,1000)
# kernel=np.random.rand(3,3)

# print(py_conv2d_multi_channel(img,kernel))
# print(py_matmul(img,kernel))
# print(np.dot(img,kernel))

import timeit

setup = '''
import numpy as np
from neural_cpp_link import py_conv2d_multi_channel
img=np.random.rand(100,100,3)
kernel=np.random.rand(3,3,3)


def conv2d(Input,kernel,padding='valid',strides=1,pad_size=1):
        multichannel=False
        if(Input.ndim==2 or kernel.ndim==2):
            xKernShape,yKernShape=kernel.shape
            xInput,yInput=Input.shape
            nChannels=1
            nfilters=1
        else:
            xKernShape,yKernShape,nfilters=kernel.shape
            xInput,yInput,nChannels=Input.shape
            multichannel=True

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

''' 

test1 = '''
print(py_conv2d_multi_channel(img,kernel,padding="valid",strides=1,pad_size=1))
'''
test2 = '''
print(conv2d(img,kernel))
'''

print( timeit.timeit(test1, setup, number=1) )
print( timeit.timeit(test2, setup, number=1) )