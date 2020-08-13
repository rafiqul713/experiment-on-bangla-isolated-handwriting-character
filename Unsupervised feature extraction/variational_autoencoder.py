from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.models import Model
import os
from sklearn import cluster,datasets
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
import glob


Images = []
Images1 = []
imageTitleList = []
#Image Loading
def loading_data(path,extention="*.png"):

    allImage = glob.glob(path + extention)

    for image in allImage:
        img=io.imread(image)
        imgName = os.path.basename(image)
        imgTitle = os.path.splitext(imgName)[0]
        imageTitleList.append(imgTitle)
        Images1.append(img)
        img=resize(img,(28,28))
        Images.append(img)

    x_train=np.array(Images)
    return x_train



# It will return  z_mean + z_log_var
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

#Calculating loss
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss




batch_size = 23
original_dim = 784 #input image 28 X 28
latent_dim = 2
intermediate_dim = 256
nb_epoch = 50
epsilon_std = 1.0

x = Input(shape=(784,))
h = Dense(256, activation='relu',name='one')(x)
z_mean = Dense(latent_dim,name='zmean')(h)
z_log_var = Dense(latent_dim,name='zLog')(h)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_h = Dense(intermediate_dim, activation='relu',name='decoder_h')
decoder_mean = Dense(original_dim, activation='sigmoid',name='decoder_mean')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)




TrainingPath="/home/rafiqul/501-600/1/"
x_train=loading_data(TrainingPath)
x_train=x_train.reshape(len(x_train),np.prod(x_train.shape[1:]))
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_train, x_train))

FEATURES= Model(vae.input,vae.get_layer('decoder_h').output)
FEATURES.compile(optimizer='rmsprop',loss=vae_loss)
output=FEATURES.predict(x_train,batch_size=batch_size)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
encoder.compile(optimizer='rmsprop', loss='mse')
x_test_encoded = encoder.predict(x_train, batch_size=batch_size)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
generator.compile(optimizer='rmsprop', loss='mse')

#Because the VAE is a generative model, we can also use it to generate new digits!
#Here we will scan the latent plane, sampling latent points at regular intervals,
#and generating the corresponding digit for each of these points.
#This gives us a visualization of the latent manifold that "generates"
#the BANGLA Alphabet.


n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()


clf=cluster.KMeans(n_clusters=5)
clf.fit(output)
labels=clf.labels_


centroid=clf.cluster_centers_
#print(centroid)
labels=clf.labels_
print(labels)
Images=np.array(Images)
imageTitleList=np.array(imageTitleList)
path="/home/rafiqul/Desktop/DataSet Bangla/TestClusteringWithConvFeature/"
for i in range(len(Images)):
    if labels[i]== 0:
        io.imsave(path+"0/"+str(imageTitleList[i]+".png"),Images1[i])
    elif labels[i] == 1:
        io.imsave(path+"1/"+str(imageTitleList[i]+".png"),Images1[i])
    elif labels[i] == 2:
        io.imsave(path+"2/"+str(imageTitleList[i]+".png"),Images1[i])
    elif labels[i] == 3:
        io.imsave(path+"3/"+str(imageTitleList[i]+".png"),Images1[i])
    elif labels[i] == 4:
        io.imsave(path+"4/"+str(imageTitleList[i]+".png"),Images1[i])


