# import the opencv library
import cv2, dlib
from facePoints import facePoints

# define a video capture object
vid = cv2.VideoCapture(0)

def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
    with open(fileName, 'w') as f:
        for p in faceLandmarks.parts():
            f.write("%s %s\n" % (int(p.x), int(p.y)))

    f.close()

# location of the model (path of the model).
Model_PATH = "/home/albert/Downloads/shape_predictor_68_face_landmarks.dat"

# now from the dlib we are extracting the method get_frontal_face_detector()
# and assign that object result to frontalFaceDetector to detect face from the image with
# the help of the 68_face_landmarks.dat model
frontalFaceDetector = dlib.get_frontal_face_detector()

# Now the dlip shape_predictor class will take model and with the help of that, it will show
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

# List to store landmarks of all detected faces
allFacesLandmark = []

while (True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    allFaces = frontalFaceDetector(imageRGB, 0)

    for k in range(0, len(allFaces)):
        # dlib rectangle class will detecting face so that landmark can apply inside of that area
        faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                           int(allFaces[k].right()), int(allFaces[k].bottom()))

        # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)

        print(detectedLandmarks.part(49) - detectedLandmarks.part(62))

        # Svaing the landmark one by one to the output folder
        allFacesLandmark.append(detectedLandmarks)

        # Now finally we drawing landmarks on face
        facePoints(frame, detectedLandmarks)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
