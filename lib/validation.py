import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import scipy.stats
import lib.model as model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Reshape
from tensorflow.keras.optimizers import SGD

def nLL(x, recon, mu, logvar, z):
  sigma = np.exp(0.5*logvar)
  b = x.shape[0]

  cross_entropy = keras.losses.binary_crossentropy(x, recon)
  log_p_x_z = -np.sum(cross_entropy,(1,2))     
  log_p_z = -np.sum(z**2/2+np.log(2*np.pi)/2,1)
  z_eps = (z - mu)/sigma
  log_q_z_x = -np.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)        
  weights = log_p_x_z+log_p_z-log_q_z_x
        
  return weights
  
def nLL_mc(x, vae, M):
  mu, logvar, _ = vae.encoder.predict(x)
  weights = 0
  for i in range(M):
    z = model.Sampling(mu, logvar)
    recon = vae.decoder.predict(z)
    sigma = np.exp(0.5*logvar)
    b = x.shape[0]

    cross_entropy = keras.losses.binary_crossentropy(x, recon)
    log_p_x_z = -np.sum(cross_entropy,(1,2))     
    log_p_z = -np.sum(z**2/2+np.log(2*np.pi)/2,1)
    z_eps = (z - mu)/sigma
    log_q_z_x = -np.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)        
    weights += log_p_x_z+log_p_z-log_q_z_x
  
  weights = weights/M      
  return weights
  

def plot_hist_s (x, vae):
  #mu, logvar, z= vae.encoder.predict(x)
  #recon = vae.decoder.predict(z)
  #weights = nLL(x, recon, mu, logvar, z)
  weights = nLL_mc(x, vae, 50)
  
  plt.figure(figsize=(12, 10))
  plt.hist(weights, alpha=0.5)
  if not (os.path.isdir('figures')):
    os.makedirs('figures')
  plt.savefig('figures/plot_hist_s.png', dpi=100)
  plt.close()

  
def plot_hist (x_in, x_out, vae):
  #mu, logvar, z= vae.encoder.predict(x_in)
  #recon = vae.decoder.predict(z)
  #mu_f, logvar_f, z_f= vae.encoder.predict(x_out)
  #recon_f = vae.decoder.predict(z_f)
  
  #weights = nLL(x_in, recon, mu, logvar, z)
  #weights_f = nLL(x_out, recon_f, mu_f, logvar_f, z_f)
  weights = nLL_mc(x_in, vae, 50)
  weights_f = nLL_mc(x_out, vae, 50)

  plt.figure(figsize=(12, 10))
  plt.hist(weights_f, alpha=0.5, label="OoD")
  plt.hist(weights, alpha=0.5, label="In-distrubution")
  plt.legend()
  plt.xlabel("log likelihood")
  
  mean = np.mean(weights)
  cov = np.sqrt(np.cov(weights))
  x = np.linspace(weights.min(0),weights.max(),1000)
  p = scipy.stats.norm(mean, cov).pdf(x)
  mean_f = np.mean(weights_f)
  cov_f = np.sqrt(np.cov(weights_f))
  x_f = np.linspace(weights_f.min(0),weights_f.max(),1000)
  p_f = scipy.stats.norm(mean_f, cov_f).pdf(x_f)
  plt.plot(x_f,p_f,label="OoD")
  plt.plot(x,p,label="In-distribution")
  
  
  

  if not (os.path.isdir('figures')):
    os.makedirs('figures')
  plt.savefig('figures/plot_hist.png', dpi=100)
  plt.close()


  
def plot_label_clusters(vae, data, labels):
  # display a 2D plot of the digit classes in the latent space
  z_mean, _, _ = vae.encoder.predict(data)
  plt.figure(figsize=(12, 10))
  plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
  plt.colorbar()
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.grid()
  #plt.show()
  if not (os.path.isdir('figures')):
    os.makedirs('figures')
  plt.savefig('figures/plot_label_clusters.png', dpi=100)
  plt.close()
  
def plot_norm(x_in, x_out,vae):
  #mu, logvar, z= vae.encoder.predict(x_in)
  #recon = vae.decoder.predict(z)
  #weights = nLL(x_in, recon, mu, logvar, z)
  weights = nLL_mc(x_in, vae, 50)

  mean = np.mean(weights)
  cov = np.sqrt(np.cov(weights))
  x = np.linspace(weights.min(0),weights.max(),1000)
  p = scipy.stats.norm(mean, cov).pdf(x)

  #mu_f, logvar_f, z_f= vae.encoder.predict(x_out)
  #recon_f = vae.decoder.predict(z_f)
  #weights_f = nLL(x_out, recon_f, mu_f, logvar_f, z_f)
  weights_f = nLL_mc(x_out, vae, 50)

  mean_f = np.mean(weights_f)
  cov_f = np.sqrt(np.cov(weights_f))
  x_f = np.linspace(weights_f.min(0),weights_f.max(),1000)
  p_f = scipy.stats.norm(mean_f, cov_f).pdf(x_f)

  plt.figure(figsize=(12, 10))
  plt.plot(x_f,p_f,label="OoD")
  plt.plot(x,p,label="In-distribution")
  plt.legend()   
  plt.xlabel("log likelihood")

  if not (os.path.isdir('figures')):
    os.makedirs('figures')
  plt.savefig('figures/plot_norm.png', dpi=100)
  plt.close()
  
def predict(x,vae,t):
  out = np.zeros(x.shape[0])
  #mu, logvar, z= vae.encoder.predict(x)
  #recon = vae.decoder.predict(z)
  #weights = nLL(x, recon, mu, logvar, z)
  weights = nLL_mc(x, vae, 50)
  out = weights >-500
  return out.astype(int)

def accurcy (xn_test, vae, yn_test, t):
  yn_predict = predict(xn_test,vae,t)
  return sum(yn_test == yn_predict)/yn_test.shape[0]
  


def VA_spec(X_train, y_train):
  ### Encoder
  encoder = Sequential()
  encoder.add(Flatten(input_shape=[28,28]))
  encoder.add(Dense(400,activation="relu"))
  encoder.add(Dense(200,activation="relu"))
  encoder.add(Dense(100,activation="relu"))
  encoder.add(Dense(50,activation="relu"))
  encoder.add(Dense(2,activation="relu"))
  
  
  ### Decoder
  decoder = Sequential()
  decoder.add(Dense(50,input_shape=[2],activation='relu'))
  decoder.add(Dense(100,activation='relu'))
  decoder.add(Dense(200,activation='relu'))
  decoder.add(Dense(400,activation='relu'))
  decoder.add(Dense(28 * 28, activation="relu"))
  decoder.add(Reshape([28, 28]))
  
  ### Autoencoder
  autoencoder = Sequential([encoder,decoder])
  autoencoder.compile(loss="mse")
  autoencoder.fit(X_train,X_train,epochs=50)
  
  
  encoded_2dim = encoder.predict(X_train)
  
  
  # create the scatter plot
  fig, ax = plt.subplots(figsize=(16,11))
  scatter = ax.scatter(
      x=encoded_2dim[:,0], 
      y=encoded_2dim[:,1], 
      c=y_train, 
      cmap=plt.cm.get_cmap('Spectral'), 
      alpha=0.4)

  # produce a legend with the colors from the scatter
  legend = ax.legend(*scatter.legend_elements(), title="Classes",bbox_to_anchor=(1.05, 1), loc='upper left',)
  ax.add_artist(legend)
  ax.set_title("Autoencoder visualization")
  plt.xlabel("X1")
  plt.ylabel("X2")
  if not (os.path.isdir('figures')):
    os.makedirs('figures')
  plt.savefig('figures/VA_spec.png', dpi=100)
  plt.close()
  
  
def display_recon(x_in, x_out, vae, n):
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # Display original
      ax = plt.subplot(3, n, i + 1)
      #plt.imshow(x_test[i].reshape(28, 28))
      plt.imshow(x_in[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # Display reconstruction from in distribution
      mu, logvar, z= vae.encoder.predict(x_in[i:i+1])
      recon = vae.decoder.predict(z)
      ax = plt.subplot(3, n, i + 1 + n+n)
      plt.imshow(recon.reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      
      # Display reconstruction from in distribution
      mu_f, logvar_f, z_f= vae.encoder.predict(x_out[i:i+1])
      recon_f = vae.decoder.predict(z_f)
      ax = plt.subplot(3, n, i + 1 + n)
      plt.imshow(recon_f.reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  #plt.show()
  if not (os.path.isdir('figures')):
    os.makedirs('figures')
  plt.savefig('figures/reconstruction.png', dpi=100)
  plt.close()