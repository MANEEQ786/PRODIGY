from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

#using pre trained model
model = InceptionV3(weights='imagenet')

img_path = 'pizza.jpeg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Use the model to predict the food item
predictions = model.predict(x)
food_label = decode_predictions(predictions, top=1)[0][0][1]

print(f'Predicted food item: {food_label}')

