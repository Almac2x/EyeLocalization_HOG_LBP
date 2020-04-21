# import the necessary packages
from skimage import feature
import numpy as np
import time
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radiusx
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, image_name,eps=1e-7):
		#Initalizes Time to show how long it computes
		Start_Time = time.time()


		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# Initalizes Time to show how long it computes
		Time_Compute = time.time() - Start_Time
		#print("--- {}s seconds to convert {} to LBP ---".format(Time_Compute,image_name))
		# return the histogram of Local Binary Patterns
		return hist