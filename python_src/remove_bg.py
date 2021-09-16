import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_detection = mp.solutions.face_detection

# For webcam input:
BG_COLOR = (192, 192, 192) # gray
cap = cv2.VideoCapture(0)

def getAbsoluteBoundingBox(relBox, windowW, windowH):
	relX = relBox.xmin
	relY = relBox.ymin
	relW = relBox.width
	relH = relBox.height

	xLeft = relX * windowW
	yLeft = relY * windowH
	
	xRight = xLeft + (relW * windowW)
	yRight = yLeft + (relH * windowH)
	
	return [xLeft, yLeft, xRight, yRight]

def expandBoundingBox(absoluteBoundingBox, scalar):
	xLeft, yLeft, xRight, yRight = absoluteBoundingBox
	w = xRight - xLeft
	h = yRight - yLeft
	wExpansion = w * scalar
	hExpansion = h * scalar
	newXleft = xLeft - wExpansion
	newYleft  = yLeft - hExpansion
	newXright = xRight + wExpansion
	newYright = yRight + hExpansion
	return [newXleft, newYleft, newXright, newYright]


with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
	with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
		bg_image = None
		while cap.isOpened():
			success, image = cap.read()
			if not success:
			  print("Ignoring empty camera frame.")
			  # If loading a video, use 'break' instead of 'continue'.
			  continue
			W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
			H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
			
			# Flip the image horizontally for a later selfie-view display, and convert
			# the BGR image to RGB.
			image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			results = selfie_segmentation.process(image)
			face_results = face_detection.process(image)

			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# Draw selfie segmentation on the background image.
			# To improve segmentation around boundaries, consider applying a joint
			# bilateral filter to "results.segmentation_mask" with "image".
			#condition = np.dstack((results.segmentation_mask,results.segmentation_mask,results.segmentation_mask))
			
			mask = results.segmentation_mask
			# drop all unused channels, this has to be done cause medipipe version for jetson is 3.5
			# should be fixed and work with the original mediapipe example with newer version on linux
			mask = np.delete(mask, 1, 2);
			mask = np.delete(mask, 1, 2);
			mask = np.delete(mask, 1, 2);
			mask = mask > 0.6
			mask = np.dstack((mask, mask, mask))
			
			
			# The background can be customized.
			#   a) Load an image (with the same width and height of the input image) to
			#      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
			#   b) Blur the input image by applying image filtering, e.g.,
			#      bg_image = cv2.GaussianBlur(image,(55,55),0)
			if bg_image is None:
			  bg_image = np.zeros(image.shape, dtype=np.uint8)
			  bg_image[:] = BG_COLOR
			 
			output_image = np.where(mask, image, bg_image)
				
			if face_results.detections:
				for detection in face_results.detections:
					rel_box = detection.location_data.relative_bounding_box
					abs_box = getAbsoluteBoundingBox(rel_box, W, H)
					xL, yL, xR, yR = expandBoundingBox(abs_box, 0.25)
					cv2.rectangle(output_image, (int(xL), int(yL)), (int(xR), int(yR)), (0, 255, 0), 2)
					
			cv2.imshow('MediaPipe Selfie Segmentation', output_image)

			if cv2.waitKey(5) & 0xFF == 27:
			  break
		  
	cap.release()
