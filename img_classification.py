def teachable_machine_classification(img, weights_file):

    import keras
    from tensorflow.keras.utils import load_img, img_to_array
    from PIL import Image, ImageOps
    import numpy as np
    
    # Load the model
    model = keras.models.load_model(weights_file)
    
    #Load the image
    #image = load_img(img,
                     #color_mode='grayscale',
                     #target_size=(128,128))
    
    # Construct the tensor that .predict is expecting
    image = ImageOps.fit(img, (128,128), Image.ANTIALIAS)
    image = ImageOps.grayscale(image) # grayscale only images
    image = img_to_array(image) # convert image to array
    data = np.expand_dims(image, axis=0)#/255 # normalize

    # Create the array of the right shape to feed into the keras model
    #data = np.ndarray(shape=(1, 128, 128, 1), dtype=np.float32)
    #image = img
    # image sizing
    #size = (128, 128)
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    #image_array = np.asarray(image)
    # Normalize the image
    #normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    #data[0] = normalized_image_array

    # get the prediction
    prediction = model.predict(data, verbose=0)
    return prediction #np.argmax(prediction) # return position of the highest probability
