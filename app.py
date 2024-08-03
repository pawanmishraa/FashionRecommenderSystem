import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None
images_dir = 'images'
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Directory '{images_dir}' does not exist.")
filenames = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
feature_list = []
for file in tqdm(filenames):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)
