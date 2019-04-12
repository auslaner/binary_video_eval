import argparse
import json
import os
import time

import cv2
import mxnet as mx
import numpy as np
import progressbar
from imutils.video import FileVideoStream

from preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from preprocessing.mean_preprocessor import MeanPreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--means", required=True,
                help="path to image channels means file associated with training data")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
                help="epoch number to load")
ap.add_argument("-o", "--output", required=True,
                help="path to video output")
ap.add_argument("-v", "--video", required=True,
                help="path to video file for model evaluation")
ap.add_argument("-sf", "--start", type=int, default=0,
                help="starting frame to begin predictions on")
ap.add_argument("-ef", "--end", type=int, default=200000,
                help="last frame in video to evaluate. Defaults to arbitrarily large number")
ap.add_argument("-g", "--gpu", type=int, default=0,
                help="GPU device number to target")
args = vars(ap.parse_args())

# Load the RGB means for the training set
means = json.loads(open(args["means"]).read())

# Load the checkpoints from disk
print("[INFO] Loading model...")
checkpoints_path = os.path.sep.join([args["checkpoints"],
                                     args["prefix"]])
model = mx.model.FeedForward.load(checkpoints_path, args["epoch"])

# Compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(args["gpu"])],
    symbol=model.symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params
)

# Initialize the image preprocessors
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor(data_format="channels_first")

vs = FileVideoStream(args["video"]).start()
time.sleep(3)  # Give the buffer a chance to fill
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
pollinator_video = cv2.VideoWriter(os.path.sep.join([args["output"], "pollinator.avi"]), fourcc, 5, (640, 480))
not_pollinator_video = cv2.VideoWriter(os.path.sep.join([args["output"], "not_pollinator.avi"]), fourcc, 5, (640, 480))

frame = 0


# Initialize the progress bar
widgets = ["Processing {}: ".format(args["video"].split("/")[-1]), progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=args["end"],
                               widgets=widgets).start()

# Load the image from disk
while vs.more():
    image = vs.read()
    frame += 1

    if image is None:
        break

    if frame < args["start"]:
        # Wait a bit so we don't overrun the buffer
        time.sleep(0.1)
        continue
    elif frame >= args["end"]:
        break

    pbar.update(frame)

    orig = image.copy()

    # Preprocess image
    image = iap.preprocess(mp.preprocess(image))
    image = np.expand_dims(image, axis=0)

    # Classify the image
    pred = model.predict(image)
    pollinator, not_pollinator = pred[0]

    if pollinator > not_pollinator:
        pollinator_video.write(orig)
    else:
        not_pollinator_video.write(orig)

pollinator_video.release()
not_pollinator_video.release()

pbar.finish()

vs.stop()
