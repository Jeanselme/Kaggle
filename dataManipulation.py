"""
	Data Manipulation
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""
import pandas
import numpy as np
import scipy.misc

# Extract labels and data
def extractPicturesFromCSV(csvFile, reshape = False):
	"""
	Returns a list of matrix images, reshape to a linear shape if reshape = True
	"""
	# Reads file
	data = pandas.read_csv(csvFile, header=None).as_matrix()

	# List of images
	pictures = []

	# Changes the data
	for row in range(len(data)):
		pic = fromVecToPic(data[row])
		if reshape:
			pic = pic.flatten()
			# To avoid numpy error of shape
			pic = pic.reshape((len(pic),1))
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

# Manipulations on labels
def changeLabels(labelsArray, label):
	"""
	Transforms the labels into a binary array with class = +1 if label=i -1 otherwise
	"""
	# Copies data
	res = data.copy()
	# Changes value
	ko, ok = res != value, res == value
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
	size = labelMax - labelMin

	for label in labelsArray:
		binaryArray = np.zeros((size,1))
		binaryArray[label] = 1
		res.append(binaryArray)

	return np.array(res)


# To display images and manipulates
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
	image[:,:,0] = r.reshape((dim,dim))
	image[:,:,1] = g.reshape((dim,dim))
	image[:,:,2] = b.reshape((dim,dim))

	return image

if __name__ == '__main__':
	extractPicturesFromCSV("data/Xtr.csv")
	print(extractLabelsFromCSV("data/Ytr.csv"))
