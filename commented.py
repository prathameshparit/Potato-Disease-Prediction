
# def predict(model, img, class_names=None):
#
#     if class_names is None:
#         class_names = ['Early_Blight', 'Healthy', 'Late_Blight']
#     predictions = model.predict(img)
#
#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * (np.max(predictions[0])), 2)
#
#     return predicted_class, confidence
#
#
# def pred_and_plot_custom(img_path, model):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
#     # plt.imshow(img)
#     x = tf.keras.preprocessing.image.img_to_array(img)
#
#     x = np.expand_dims(x, axis=0)
#     predicted_class, confidence = predict(model, x)
#     images = np.vstack([x])
#     pred = model.predict(images)
#     if pred[0][0] > 0.5:
#         acutal_class = "Early_blight"
#     elif pred[0][1] > 0.5:
#         acutal_class = "Healthy"
#     elif pred[0][2] > 0.5:
#         acutal_class = "Late_blight"
#     else:
#         acutal_class = "Unknown"
#
#     return predicted_class, confidence


# img_path = "test/Early_Blight_59.jpg"
# print(pred_and_plot_custom(img_path, model))