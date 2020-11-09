from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay
from utility import *
from model import edsr


train_path = "D:/places_dataset/train_20k.pickle"
valid_path = "D:/places_dataset/valid_4k_processed.pickle"
test_path = "D:/places_dataset/test_4k_processed.pickle"
input_train_path = "D:/places_dataset/train_20k_128_processed.pickle"
input_valid_path = "D:/places_dataset/valid_4k_128_processed.pickle"
input_test_path = "D:/places_dataset/test_4k_128_processed.pickle"

print("train Y")
train_y = load_data(train_path)
# plot_data(train_y)
train_y = process_data(train_y,256)

print("Validation Y")
val_y = load_data(valid_path)
val_y = process_data(val_y,256)

print("shape of train {train} , shape of validate {valid}".format(train=train_y.shape, valid=val_y.shape))
print("max of train is {train} , max of validate {valid}".format(train=np.max(train_y), valid=np.max(val_y)))


print("loading train Images")
train_X = load_data(input_train_path)
val_y = process_data(val_y,128)
print("loading validation data")
val_X = load_data(input_valid_path)
val_y = process_data(val_y,128)

batch_size = 16
epochs = 200

autoencoder = edsr()
autoencoder.summary()

opt = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[15000], values=[3e-4, 5e-5]))
autoencoder.compile(optimizer=opt, loss='mean_absolute_error')
autoencoder_train = autoencoder.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_X, val_y))

autoencoder.save('C:/Users/Desktop/200_epochs_20k_abs.h5')

plot_results(autoencoder_train, epochs)



