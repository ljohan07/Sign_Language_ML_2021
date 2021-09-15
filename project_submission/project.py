#!/usr/bin/env python3

import cv2
from tensorflow import keras

# Helper function
def find_class(label):
    letter_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y'}
    max_val = max(label)

    for index, elem in enumerate(label):
        if elem == max_val:
            return letter_dict[index]


# Main function
if __name__ == '__main__':

    # Load model
    model = keras.models.load_model('95_model')
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    # Live video
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Red rectangle and crop
        dimensions = image.shape
        height, width = dimensions
        crop1 = 0
        crop2 = 0
        if height > width:
      	    crop1 = (height - width) / 2
      	    crop2 = height - crop1
      	    cv2.rectangle(frame, (0, int(crop1)), (width, int(crop2)), (0, 0, 255), 5)
      	    image = image[int(crop1):int(crop2), 0:width]
        elif width > height:
            crop1 = (width - height) / 2
            crop2 = width - crop1
            cv2.rectangle(frame, (int(crop1), 0), (int(crop2), height), (0, 0, 255), 5)
            image = image[0:height, int(crop1):int(crop2)]
            cv2.imshow('Live Webcam', frame)

    	# Resize and reshape
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, 28, 28, 1)

    	# Make prediction
        y_pred = model.predict(image)
        print(find_class(y_pred[0]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
      	    break
