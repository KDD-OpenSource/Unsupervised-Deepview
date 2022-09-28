import torch
from torchvision import datasets
from torchvision import transforms
import PyQt5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from deepview import DeepView
from deepview_unsupervised.DeepView import DeepView_unsupervised
#from deepview_original.DeepView import DeepView_original

import numpy as np
import time
# ---------------------------
import demo_utils as demo
import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot
import tensorflow as tf
from scipy.spatial.distance import pdist, cdist, squareform
from keras.models import load_model
from keras.datasets import mnist, cifar10, fashion_mnist

softmax = torch.nn.Softmax(dim=-1)
#
# #device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_model = demo.create_torch_model(device)
model = load_model('data/model_cifar.h5')
model_fashion_mnist = load_model('mymodel/')

# this is the prediction wrapper, it encapsulates the call to the model
# and does all the casting to the appropriate datatypes
# def pred_wrapper(x):
#     with torch.no_grad():
#         x = np.array(x, dtype=np.float32)
#         tensor = torch.from_numpy(x).to(device) #keras doesn t need
#         logits = torch_model(tensor)
#         probabilities = softmax(logits).cpu().numpy()
#     return probabilities

def pred_wrapper(x): ###### keras predwrapper
    x = np.array(x, dtype=np.float32)
    logits= model_fashion_mnist(x)
    probabilities = tf.nn.softmax(logits).cpu()
    return probabilities



def visualization(image, point2d, pred, label=None, title=None):
    f, a = plt.subplots()
    a.set_title(title)
    a.imshow(image.transpose([1, 2, 0]))


# --- Deep View Parameters ----
batch_size = 32
max_samples = 1024
data_shape = (28, 28)
n = 5
lam = .65
resolution = 100
cmap = 'tab10'
title = 'fashion-MNIST'

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes= ('tshirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
#classes = ('uncertain', 'certain') # rausnehmen für unsupervised

#deepview_original = DeepView_original(pred_wrapper, classes, max_samples, batch_size, data_shape, n, lam, resolution, cmap, title=title, data_viz=None)

deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                    data_shape, n, lam, resolution, cmap, title=title, data_viz=None)
deepview_unsup = DeepView_unsupervised(pred_wrapper, max_samples, batch_size,
                    data_shape, n, lam, resolution, cmap, title=title, data_viz=None)

#deepview_unsup = DeepView_unsupervised(pred_wrapper, max_samples, batch_size, data_shape, n, lam, resolution, cmap, title=title, data_viz=None)


######################Keep in mind Umap bad for CIFAR 10 alone, see TODO https://github.com/jlmelville/uwot/blob/master/docs/umap-examples.md
#get CIFAR-10 data
testset = demo.make_cifar_dataset()
#plt.imshow(np.array(testset[8][0]).T) #how testset images are accessible
#plt.show()
n_samples = 1024
sample_ids = np.random.choice(len(testset), n_samples)

#sample_ids = np.random.choice(len(testset), n_samples)
#testset = cifar10.load_data() TODO
sample_ids = np.random.choice(len(testset), n_samples)
X = np.array([ testset[i][0].numpy() for i in sample_ids ])
Y = np.array([ testset[i][1] for i in sample_ids ])

path = "data/"
X_adv= np.load(path + "Adv_cifar_cw-l2.npy")
X_adv =np.array([X_adv[i] for i in range(max_samples)])
X_adv.reshape(max_samples, 32, 32, 3) #adjust to data shape
#print(X_adv.shape)
Y_adv= np.load(path + "Adv_labels_cifar_cw-l2.npy")
Y_adv = np.argmax(Y_adv, axis=1)
Y_adv =np.array([Y_adv[j] for j in range(max_samples)])

#--------------------------
#deepview
t0 = time.time()

##########load fashion_mnist
# load dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
sample_ids= np.random.choice(len(trainX), n_samples)
X_train = np.array([ trainX[i] for i in sample_ids ])
Y_train= np.array([ trainy[i] for i in sample_ids ])
print(X_train.shape)
print(Y_train.shape)
#plt.imshow(trainX[2],cmap=pyplot.get_cmap('gray'))
#plt.show()
#plt.imshow(X_adv[12], cmap=pyplot.get_cmap('gray'))
#plt.show()

#------------------------------------------
#deepview_original.add_samples(X,Y)
#deepview_original.show()

#-------------------------------------------
#deepview.add_samples(X,Y)
#deepview.show()
#plt.close('all')
deepview_unsup.add_samples(X_train) #, Y_train)
deepview_unsup.show()
#plt.close('all')
