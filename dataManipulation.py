"""
	Data Manipulation
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""
import pandas
import collections
import numpy as np
import scipy.misc
from scipy.ndimage import rotate
from utils.imageDescriptor import *

# Extract labels and data
def extractPicturesFromCSV(csvFile):
	"""
	Returns a list of matrix images
	"""
	# Reads file
	data = pandas.read_csv(csvFile, header=None).as_matrix()

	# List of images
	pictures = []

	# Changes the data
	for row in range(len(data)):
		pic = fromVecToPic(data[row])
		pictures.append(pic)

	return np.array(pictures)

def extractLabelsFromCSV(csvFile):
	"""
	Returns a list of labels
	"""
	return pandas.read_csv(csvFile)["Prediction"].as_matrix()

def addRotations(images, labels, rotation=[-30,-15,-10,-5,5,10,15,30]):
	"""
	Returns the list of initial images with different rotations of these images
	"""
	imagesRes = []
	labelsRes = []
	for image, label in zip(images, labels):
		imagesRes.append(image)
		labelsRes.append(label)
		for angle in rotation:
			imagesRes.append(rotate(image, angle, reshape=False))
			labelsRes.append(label)

	return np.array(imagesRes), np.array(labelsRes)

def saveLabelsToCsv(labelsArray, fileName):
	"""
	Saves labels in the csvFile
	"""
	# Creates an adaptated dataFrame for Kaggle
	df = pandas.DataFrame()
	df['Id'] = range(1,len(labelsArray)+1)
	df['Prediction'] = labelsArray

	# Saves the data
	df.to_csv(fileName, index = False)

def saveImagesToCsv(imagesArray, fileName):
	"""
	Saves images in the csvFile
	"""
	flattenImage = [image.flatten() for image in imagesArray]
	df = pandas.DataFrame(flattenImage)
	df.to_csv(fileName, index = False)

def shuffleDataLabel(dataArray, labelsArray):
	"""
	Shuffles data and label
	"""
	permut = np.random.permutation(len(labelsArray))
	return dataArray[permut], labelsArray[permut]

def balance(dataArray, labelsArray, shuffle):
	"""
	Creates a partial database with same number of each category
	"""
	if shuffle:
		dataArray, labelsArray = shuffleDataLabel(dataArray, labelsArray)
	selection = np.array([], dtype=int)
	count = collections.Counter(labelsArray)
	minLabel = min(count.values())
	for label in count.keys():
		selection = np.append(selection, np.where(labelsArray == label)[:minLabel])
	return shuffleDataLabel(dataArray[selection], labelsArray[selection])

def selectLabels(dataArray, labelsArray, labels):
	"""
	Transforms the labels into a binary array with class = +1 if label=i, -1 if label=j
	"""

	selection = np.array([], dtype=int)
	for label in labels:
		selection = np.append(selection, np.where(labelsArray == label))
	return shuffleDataLabel(dataArray[selection].copy(), labelsArray[selection].copy())

# Manipulations on labels
def changeLabels(labelsArray, label):
	"""
	Transforms the labels into a binary array with class = +1 if label=i -1 otherwise
	"""
	# Copies data
	res = labelsArray.copy()
	# Changes value
	ko, ok = res != label, res == label
	res[ko],res[ok] = -1, 1
	return res

def binaryArrayFromLabel(labelsArray):
	"""
	Transforms the array into array of array with
	"""
	res = []
	# Computes the size of the label array
	labelMax = np.max(labelsArray)
	labelMin = np.min(labelsArray)
	size = labelMax - labelMin + 1

	for label in labelsArray:
		binaryArray = np.zeros((size,1))
		binaryArray[label] = 1
		res.append(binaryArray)

	return np.array(res)

# Manipulation on images
def describeImages(imagesArray):
	"""
	Flatten images into 1D array
	And add the histogram of gradient and of color
	"""
	res = []

	for image in imagesArray:
		alpha, intensity = hOg(image)
		r, g, b = hOc(image)
		localAlpha, localIntensity, localR, localG, localB = localDescriptor(image)
		resImage = np.concatenate((image.flatten(), alpha, intensity, r, g, b, localAlpha, localIntensity, localR, localG, localB))
		resImage = resImage.reshape((len(resImage),1))
		res.append(resImage)

	return np.array(res)

def displayPicture(image):
	scipy.misc.imshow(image)

def fromVecToPic(vectorRGB):
	"""
	Transforms a vector of pixels of format
	red pixels, green pixels, blue pixels
	into a matrix of size sqrt(len(red pixels))^2
	"""
	third = int(len(vectorRGB)/3)

	# Separates different color
	r = np.array(vectorRGB[0:third])
	g = np.array(vectorRGB[third:2*third])
	b = np.array(vectorRGB[2*third:3*third])

	# Computes image
	dim = int(np.sqrt(third))

	image = np.zeros((dim,dim,3))
	image[:,:,0] = b.reshape((dim,dim))
	image[:,:,1] = b.reshape((dim,dim))
	image[:,:,2] = r.reshape((dim,dim))

	return image

if __name__ == '__main__':
	extractPicturesFromCSV("data/Xtr.csv")
	print(extractLabelsFromCSV("data/Ytr.csv"))
