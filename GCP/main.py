from PIL import Image
import base64
import io
import numpy as np
import cv2
import tensorflow as tf


#Place your model
#=============================
model = tf.keras.models.load_model('SelectedModel.h5')
#=============================

def predict(image_np):

    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    imageReshape = image[None, :]


    score = model.predict(imageReshape)[:, 3]
    return score[0]


def handler(request):
    #CORS headers
    if request.method == 'OPTIONS':

        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

 
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    request_json = request.get_json()
    base64_encoded_image = request_json['image']
    base64_decoded = base64.b64decode(base64_encoded_image)
    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)


    probability = predict(image_np)

    print(probability)
    prob_str = str(probability)
    return (prob_str, 200, headers)



