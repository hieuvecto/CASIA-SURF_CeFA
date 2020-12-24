# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.face_utils import shape_to_np
import argparse
import imutils
import dlib
import cv2
import time
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
from at_learner_core.predictor import Predictor
from models.wrappers.multi_modal_wrapper import MultiModalWrapper

# Define predictor class
class SinglePredictor(Predictor):
	def __init__(self, test_config, model_config, checkpoint_path):
		self.test_config = test_config
		self.model_config = model_config
		self.device = torch.device("cuda" if self.test_config.ngpu else "cpu")
		checkpoint = torch.load(checkpoint_path, map_location='cpu')

		self._init_wrapper(checkpoint)
		self.transforms = self.model_config.datalist_config.testlist_configs.transforms

	def transform_data(self, data):
		if self.transforms is not None:
			data = self.transforms(data)

		data['video_id'] = 'testvideo01'

		for key,value in data.items():
			if isinstance(value, torch.Tensor):
				data[key] = torch.unsqueeze(value, 0)
			else:
				data[key] = [value]

		# redefine target
		data['target'] = torch.Tensor([0.])

		return data

	def run_predict(self, data, thr = 0.5):
		data = self.transform_data(data)
		
		# for key,value in data.items():
		# 	if isinstance(value, torch.Tensor):
		# 		print("KEY")
		# 		print(key)
		# 		print("VALUE SHAPE")
		# 		print(value.shape)

		# exit()

		self.wrapper.eval()
		with torch.no_grad():
			if isinstance(data, dict) or isinstance(data, OrderedDict):
				for k, v in data.items():
					if isinstance(v, torch.Tensor):
						data[k] = v.to(self.device)
			else:
				data = data.to(self.device)

			output_dict, _ = self.wrapper(data)

			output = output_dict['output']
			# output[output < thr] = 0
			# output[output >= thr] = 1
			# output.astype(np.int64)

			print("OUTPUT")
			print(output)

	def _init_wrapper(self, checkpoint):
		self.wrapper = MultiModalWrapper(self.model_config.wrapper_config)
		self.wrapper.load_state_dict(checkpoint['state_dict'])
		self.wrapper = self.wrapper.to(self.device)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, type=str,
				help="path to facial landmark predictor")
ap.add_argument('--test_config_path',
					type=str,
					help='Path to test config')
ap.add_argument('--model_config_path',
					type=str,
					help='Path to model config')
ap.add_argument('--checkpoint_path',
					type=str,
					help='Path to checkpoint')
args = ap.parse_args()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor)
fa = FaceAligner(predictor, desiredFaceWidth=256)

# start the video stream thread
print("[INFO] starting video stream thread...")
# define a video capture object 
vs = cv2.VideoCapture(0) 
# make 144p
vs.set(3, 80)
vs.set(4, 80)

time.sleep(1.0)

# anti spoofing vars
frame_array = []
MAX_FRAME = 30


# torch load config 
test_config = torch.load(args.test_config_path)
model_config = torch.load(args.model_config_path)

paper_predictor = SinglePredictor(test_config, model_config, args.checkpoint_path)

#init timer
timer = None

# loop over frames from the video stream
while True:
	if len(frame_array) == 0:
		timer = time.time()

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	ret, frame = vs.read()
	
	# Resize
	frame = imutils.resize(frame, width=800)

	# Convert it to HSV
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 1)
	
	# show the frame
	cv2.imshow("Frame", frame)

	# loop over the face detections
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks

		# TODO: add try catch
		(x, y, w, h) = rect_to_bb(rect)
		w_ratio = 1.1
		h_ratio = 1.2
		new_w = round(w * w_ratio)
		new_h = round(h * h_ratio)

		temp_x = x - round((new_w - w) / 2)
		new_x = temp_x if temp_x > 0 else 0
		temp_y = y - round((new_h - h) / 2)
		new_y = temp_y if temp_y > 0 else 0

		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		mask = np.zeros(frame.shape, dtype=np.uint8)
		roi_corners = []

		channel_count = frame.shape[2]
		ignore_mask_color = (255,)*channel_count

		for i in range(17):
			roi_corners.append((shape[i,0], shape[i,1]))

		peak = round(shape[0,1] - 2.3 * (shape[0,1]- shape[19,1]) )
		peak = peak if peak > 0 else 0
		semiPeak = round(shape[0,1] - 1.8 * (shape[0,1]- shape[19,1]) )
		semiPeak = semiPeak if semiPeak > 0 else 0

		leftSemiX = round(shape[0,0] + (shape[17,0] - shape[0,0]) / 2)
		leftSemiX = leftSemiX if leftSemiX > 0 else 0
		rightSemiX = round(shape[26,0] + (shape[16,0] - shape[26,0]) / 2)
		rightSemiX = rightSemiX if rightSemiX > 0 else 0

		roi_corners.append((rightSemiX, semiPeak))
		roi_corners.append((shape[24,0], peak))
		roi_corners.append((shape[17,0], peak))
		roi_corners.append((leftSemiX, semiPeak))

		roi_corners = np.array([roi_corners], dtype=np.int32)

		cv2.fillPoly(mask, roi_corners, ignore_mask_color)

		masked_frame = cv2.bitwise_and(frame, mask)
		masked_face = imutils.resize(masked_frame[new_y:new_y + new_h, new_x:new_x + new_w], width=280)

		cv2.imshow("Masked face", masked_face)
		# print(type(masked_face))
		# print(masked_face.shape)

		# prepare model data
		pil_image = Image.fromarray(masked_face)
		frame_array.append(pil_image)

		if len(frame_array) >= MAX_FRAME:
			item_dict = OrderedDict()

			item_dict['data'] = frame_array
			item_dict['target'] = torch.Tensor([0.])

			paper_predictor.run_predict(item_dict)

			predict_timer = time.time()
			print("Predict time: {} secs".format(predict_timer - timer))
			frame_array = []

		# only show one aligned face
		break

	# cv2.imshow("Frame2", hsv_frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()