from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import sys

# Path to folders with training data
target_path = Path("training_data")
not_target_path = Path("other_data")

def feature_extract():
    images = []
    labels = []

    for img in not_target_path.glob("*.png"):
        img = image.load_img(img)
        image_array = image.img_to_array(img)
        images.append(image_array)
        labels.append(0)

    for img in target_path.glob("*.png"):
        img = image.load_img(img)
        image_array = image.img_to_array(img)
        images.append(image_array)
        labels.append(1)

    x_train = np.array(images)
    y_train = np.array(labels)

    # Normalize
    x_train = vgg16.preprocess_input(x_train)
    pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    features_x = pretrained_nn.predict(x_train)

    return (features_x, y_train )

def define_model(x_train, y_train):
    #Create model
    model = Sequential()
    model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=10,
        shuffle=True
    )

    return (model)

def predict_now(model, image_file):
    img = image.load_img(image_file, target_size=(64, 64))
    image_array = image.img_to_array(img)
    images = np.expand_dims(image_array, axis=0)
    images = vgg16.preprocess_input(images)
    feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    features = feature_extraction_model.predict(images)
    results = model.predict(features)
    single_result = results[0][0]
    print("\n\nLikelihood that this image contains target object: {}%".format(int(single_result * 100)))


#main()

if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Please provide an image file name")
        sys.exit()

    if sys.argv[1] == None:
        print ("Please provide a valid image file name")
    else:
        x_train, y_train = feature_extract()
        model = define_model(x_train, y_train)
        predict_now(model, sys.argv[1])
