import os
import numpy as np

import tensorflow as tf
from keras import models

from recording import record_audio, terminate
from audio_converting import preprocess_audiobuffer



# !! Modify this in the correct order
commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

os.chdir(r"C:\Users\s8gre\Documents\Schule\KerasProjects\VoiceRecongnition")
loaded_model = tf.saved_model.load(".\AI\data\save")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction["predictions"], axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    while True:
        command = predict_mic()
        print(command)
        if command == "stop":
            terminate()
            break