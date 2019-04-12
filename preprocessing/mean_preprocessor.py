import cv2


class MeanPreprocessor:
    def __init__(self, r_mean, g_mean, b_mean):
        # Store the red, green, and blue channel averages across
        # a training set
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        # Split the image into its respective red, green, and blue
        # channels
        (b, g, r) = cv2.split(image.astype("float32"))

        # Subtract the means for each channel
        r -= self.r_mean
        g -= self.g_mean
        b -= self.b_mean

        # merge the channels back together and return the image
        return cv2.merge([b, g, r])
