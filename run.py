import lib.load_data as ld
import lib.model as model
import lib.validation as validation
import sys
import os
import subprocess

################################
# Complete the functions below #
################################

# Download/create the dataset
def fetch():
  print("fetching dataset!")  # replace this with code to fetch the dataset
  ld.download_mnist('dataset')
  ld.download_fmnist('dataset')
  
# Train your model on the dataset
def train():
  
  print("training model!")  # replace this with code to train the model
  latent_dim = 2
  original_dim = (28, 28, 1)
  batch_size = 100
  epochs = 30
    
  #load data
  x_train, y_train, x_val, y_val, x_test, y_test = ld.load_mnist('dataset',flatten=False)
  xf_train, yf_train, xf_test, yf_test, _, _ = ld.load_fmnist('dataset',flatten=False)

  #encoder
  encoder_inputs = model.keras.Input(shape=original_dim)
  z_mean, z_log_var = model.create_encoder(encoder_inputs, latent_dim)
  z = model.Sampling(z_mean, z_log_var)
  encoder = model.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  #encoder.summary()

  #decoder
  latent_inputs = model.keras.Input(shape=(latent_dim,))
  decoder_outputs = model.create_decoder(latent_inputs, latent_dim)
  decoder = model.keras.Model(latent_inputs, decoder_outputs, name="decoder")
  #decoder.summary()

  vae = model.VAE(encoder, decoder)
  vae.compile(optimizer=model.keras.optimizers.Adam())
  vae.fit(x_train, epochs=epochs, batch_size=batch_size)
  #vae.fit(xf_train, epochs=epochs, batch_size=batch_size)
  
  #store model
  if not (os.path.isdir('param')):
    os.makedirs('param')
  vae.encoder.save("param/encoder_save", overwrite=True)
  vae.decoder.save("param/decoder_save", overwrite=True)



# Compute the evaluation metrics and figures
def evaluate():
  print("evaluating model!")
  t = -500
  #load model
  encoder = validation.keras.models.load_model("param/encoder_save")
  decoder = validation.keras.models.load_model("param/decoder_save")
  vae = model.VAE(encoder, decoder)
  vae.compile(optimizer=model.keras.optimizers.Adam())

  #load data
  _, _, _, _, x_test, y_test = ld.load_mnist('dataset',flatten=False)
  _, _, xf_test, yf_test, _, _ = ld.load_fmnist('dataset',flatten=False)
  xn_test, yn_test = ld.load_mnist_vs_fmnist('dataset',flatten=False)

  validation.plot_hist(x_in=x_test, x_out=xf_test, vae=vae)
  validation.plot_norm(x_in=x_test, x_out=xf_test, vae=vae)
  #validation.plot_hist(x_in=xf_test, x_out=x_test, vae=vae)
  #validation.plot_norm(x_in=xf_test, x_out=x_test, vae=vae)

  #validation.VA_spec(xf_test,yf_test)

  print(validation.accurcy (xn_test,vae, yn_test, t))
  

# Compile the PDF documents
def build_paper():
  print("building papers!")  # replace this with code to make the papers
  subprocess.call('pdflatex model.tex')
  os.system('start model.pdf')
  
  subprocess.call('pdflatex main.tex')
  os.system('start main.pdf')


###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
  print("""
    You need to pass in a command-line argument.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
  """)
  sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
  supported_functions[arg]()
else:
  raise ValueError("""
    '{}' not among the allowed functions.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """.format(arg))
