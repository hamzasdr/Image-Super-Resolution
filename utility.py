import os
import cv2
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from math import log10, sqrt
from skimage.measure import compare_ssim


def create_data(dir, count, img_size):
    # for category in CATEGORIES:  # do dogs and cats
    # path = os.path.join(TRAINDIR,os.listdir(TRAINDIR)[category])  # create path to dogs and cats
    # path = os.path.join(TRAINDIR,category)
    training_data = []
    for img in os.listdir(dir):
        if count == 0:
            break
        img_array = cv2.imread(os.path.join(dir, img))  # convert to array
        b, g, r = cv2.split(img_array)  # get b, g, r
        img_array = cv2.merge([r, g, b])
        new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
        training_data.append(new_array)  # add this to our training_data
        count -= 1
    return np.array(training_data).reshape((-1, img_size, img_size, 3))


def load_split_pickle(path, split_size=0.2):
    picklein = open(path, "rb")
    training_data = pickle.load(picklein)
    # training_data = np.array(training_data)
    print("data loaded with length : ", len(training_data))
    train_X, valid_X, train_ground, valid_ground = train_test_split(training_data,
                                                                    training_data,
                                                                    test_size=split_size,
                                                                    random_state=42)
    # print(train_X.shape)
    # plt.imshow(train_X[0])
    # plt.show()
    return train_X, valid_X


def save_data(path, data):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load_data(path):
    pickle_in = open(path, "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data


def process_data(data, size, ch=3):
    # data = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data]
    data = np.array(data)
    data = np.cast['float16'](data)
    data = data.reshape(-1, size, size, ch)
    max_val = np.max(data)
    # data = np.array(data,dtype= 'float16').reshape(-1,size,size,ch)/np.max(data) # or this
    return data / max_val


def plot_data(data, num=10):
    # assert (1 <= num <= 20)
    # num = min(len(data),num)
    plt.figure(figsize=(2 * num, 10))
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.title("Image: " + str(i))
        plt.imshow(data[i])
    plt.show()


def plot_results(model, epochs):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(epochs)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def psnr(path, test_data, out_data):
    model = load_model(path)
    pred = model.predict(test_data)
    pred = np.cast['float16'](pred)

    mse = np.mean((out_data - pred) ** 2)
    print("MES :  ", mse)
    if mse == 0:  # no noise
        return 100
    PSNR = 20 * log10(1.0 / sqrt(mse))
    # print("done")
    return PSNR



def add_noise(data):
    x = 0
    y = 0
    res = np.ones((data.shape[0], 256, 256, 3))
    for i in range(0, 256, 1):
        for j in range(0, 256, 1):
            if i % 2 == 0 and j % 2 == 0:
                # c+=1
                res[:,i, j] = data[:, x, y]
                y += 1
                if y == 128:
                    y = 0
                    x += 1
            else:
                res[:,i, j] = [1, 1, 1]
    return res


def get_learning_rates(num=50):
    learning_rates = []
    for i in range(num):
        val = -4 * np.random.rand()
        learning_rates.append(10 ** val)
        # print(10**val)
    return learning_rates


def predict_ssim(path, test_data, out_data):
    model = load_model(path)
    pred = model.predict(test_data)
    print(pred.dtype, out_data.dtype)
    pred = np.cast['float16'](pred)
    # out_data = np.cast['float16'](out_data)
    res = []
    for i in range(len(out_data)):
        (score, diff) = compare_ssim(out_data[i], pred[i], full=True, multichannel=True)
        res.append(score)
    return np.mean(res)

