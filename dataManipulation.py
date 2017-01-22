"""
	Data Manipulation
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""
import pandas
import collections
import numpy as np
import scipy.misc

# Extract labels and data
def extractPicturesFromCSV(csvFile, blackAndWhite = False):
	"""
	Returns a list of matrix images
	"""
	# Reads file
	data = pandas.read_csv(csvFile, header=None).as_matrix()

	# List of images
	pictures = []

	# Changes the data
	for row in range(len(data)):
		pic = fromVecToPic(data[row], blackAndWhite)
		pictures.append(pic)

	return np.array(pictures)

def extractLabelsFromCSV(csvFile):
	"""
	Returns a list of labels
	"""
	return pandas.read_csv(csvFile)["Prediction"].as_matrix()

def saveLabelsToCsv(labelsArray, fileName):
	"""
	Saves labels in the csvFile
	"""
	# Creates an adaptated dataFrame for Kaggle
	df = pandas.DataFrame()
	df['Id'] = range(1,len(labels)+1)
	df['Prediction'] = labelsArray

	# Saves the data
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
def flattenImages(imagesArray):
	"""
	Flatten images into 1D array
	"""
	res = []

	for image in imagesArray:
		image = image.copy().flatten()
		# To avoid numpy error of shape
		image = image.reshape((len(image),1))
		res.append(image)

	return np.array(res)

def displayPicture(image):
	scipy.misc.imshow(image)

def fromVecToPic(vectorRGB, blackAndWhite):
	"""
	Transforms a vector of pixels of format
	red pixels, green pixels, blue pixels
	into a matrix of size sqrt(len(red pixels))^2
	If blackAndWhite = True => Grayscale image
	"""
	third = int(len(vectorRGB)/3)

	# Separates different color
	r = np.array(vectorRGB[0:third])
	g = np.array(vectorRGB[third:2*third])
	b = np.array(vectorRGB[2*third:3*third])

	# Computes image
	dim = int(np.sqrt(third))
	if blackAndWhite:
		image = np.zeros((dim,dim))
		image = (0.2989 * r + 0.5870 * g
		 	+ 0.1140 * b).reshape((dim,dim))
	else:
		image = np.zeros((dim,dim,3))
		image[:,:,0] = r.reshape((dim,dim))
		image[:,:,1] = g.reshape((dim,dim))
		image[:,:,2] = b.reshape((dim,dim))

	return image

if __name__ == '__main__':
	extractPicturesFromCSV("data/Xtr.csv")
	print(extractLabelsFromCSV("data/Ytr.csv"))
