import tensorflow as tf
import joblib
import sklearn
# from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib as plt
import gzip




class_names = ['Early Blight', 'Healthy', 'Late Blight']
IMAGE_SHAPE = (224, 224)

# model = tf.keras.models.load_model("models/CNN_Model_2")
# KNN_model = joblib.load('models/KNN.sav')


# SVM_model = joblib.load('models/SVM.sav')
# DT_model = joblib.load('models/DTC.sav')
# ANN_model = joblib.load('models/MLP.sav')
# Hybrid_model = joblib.load('models/Hybrid.sav')


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.io.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.
    else:
        return img


def pred_and_plot(img, model_name):
    img = load_and_prep_image(img, scale=False)  # load in target image and turn it into tensor
    pred_prob = model_name.predict(
        tf.expand_dims(img, axis=0))  # make prediction on image with shape [None, 224, 224, 3]
    pred_class = class_names[pred_prob.argmax()]  # find the predicted class label
    print(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
    # # Plot the image with appropriate annotations
    # plt.figure()
    # plt.imshow(img / 255.)  # imshow() requires float inputs to be normalized
    # plt.title(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
    #
    # plt.axis(False)
    confidence = round(100 * (pred_prob.max()), 2)
    return pred_class, confidence


def prep_img(img_path):
    IMG_SIZE = (224, 224)
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, IMG_SIZE)

    return img / 255


def pred_ml_model(img_path, model_name):
    img = prep_img(img_path)
    nx, ny, nrgb = img.shape
    img = np.reshape(img, (1, nx * ny * nrgb))
    pred = model_name.predict(img)
    confidence = model_name.predict_proba(img)
    confidence = confidence.tolist()
    print(confidence[0])
    confidence = round(100 * (max(confidence[0])), 2)
    # return (f'{y_pred_prob[0, ix]:.2%}')

    pred_class = class_names[pred[0]]

    return pred_class, confidence


# img_path = "test/Late_Blight_106.jpg"
# print(pred_ml_model(img_path, KNN_model))
# print(pred_and_plot(img_path, model))
