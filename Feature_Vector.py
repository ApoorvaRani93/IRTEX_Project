#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow as tf
from keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input


# In[39]:



from tensorflow.keras.models import Sequential


# In[3]:


filename = 'C:/Users/rania/Downloads/cifar10-pngs-in-folders/cifar10/cifar10/test/bird/0068.png'


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
img = image.load_img(filename, target_size=(32, 32))
plt.imshow(img)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)


# In[105]:


base_model = ResNet50(weights='imagenet', pooling=max, include_top = False,input_shape=(32, 32, 3))
input = Input(shape=(32,32,3),name = 'img')
x = base_model(input)
print(x.shape)
x = Flatten()(x)
print(x.shape)


model = Model(inputs=input, outputs=x)


# In[102]:


#fetched properly of lastlayer(not the softmax)
features_old = model.predict(img)
features_old


# In[60]:


base_model.summary()


# In[72]:


#trying to fetch last but fourth layer(batch normalisation)
print(base_model.layers[-3].output)


# In[97]:


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
base_model = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet', pooling=max, include_top = False)
#base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
#input = Input(shape=(32,32,3),name = 'img')
model = tensorflow.keras.Sequential()
model.add(InputLayer(input_shape=(32,32,3), name='img'))
for layer in base_model.layers[0:176]:
    model.add(layer)


#intermediate_layer_model = Model(inputs=input,outputs = base_model.layers[-3].output)


# In[ ]:


features = model.predict(img)
features


# In[74]:


#from keras.models import Model
#base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
#model = base_model  # include here your original model
#input = Input(shape=(32,32,3),name = 'img')
#print(input)
#layer_name = 'layer_!5'
#m = tf.keras.Sequential()
#m.add(base_model.layers[39].output)

#intermediate_layer_model = Model(inputs=input,
#                                 outputs=model.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model.predict(data)
#model.summary()


# In[ ]:


#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
#intlayer = base_model.layers[100].output
#intlayer_reshape = tf.reshape(intlayer,shape = (1,1,1024))
#print(intlayer_reshape.get_shape())
#X_shape = tf.TensorShape([None]).concatenate(intlayer_reshape.get_shape()[0:])
#X = tf.placeholder_with_default(intlayer, shape=X_shape)
#X


# In[ ]:




