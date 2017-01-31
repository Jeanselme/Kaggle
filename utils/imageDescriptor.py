"""
	Image Descriptor
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np

# To avoid division by 0
epsilon = 0.0000001

def normalize(array):
	"""
	Normlizes the array
	"""
	return (array - np.mean(array)) / max(epsilon,np.var(array))

def histogram(flatArray, downBound, upBound, numberInterval):
	"""
	Creates the histogram of flatArray into number of interval
	"""
	intervals = max(1,(upBound - downBound)/numberInterval)
	res = [0] * numberInterval

	for data in flatArray:
		res[max(0, min(int(data - downBound// intervals), numberInterval -1))] += 1

	return np.array(res)

def hOg(image, numberInterval = 100, RGB = True):
	"""
	Computes the histogram of gradient
	"""
	bwimage = image.copy()
	if RGB:
		r = image[:,:,0]
		g = image[:,:,1]
		b = image[:,:,2]
		bwimage = (0.2989 * r + 0.5870 * g
		 	+ 0.1140 * b)

	gradx, grady = np.gradient(bwimage)
	alpha = np.arcsin(grady/np.sqrt(gradx**2 + grady**2 + epsilon))
	intensity = normalize(np.sqrt(gradx**2 + grady**2))

	alphaHistogram = histogram(alpha.flatten(), -np.pi/2, np.pi/2, numberInterval)
	intensityHistogram = histogram(intensity.flatten(), -1, 1, numberInterval)

	return normalize(alphaHistogram), normalize(intensityHistogram)

def hOc(image, numberInterval = 100):
	"""
	Computes the histogram of color of the image
	-> Need RGB image
	"""
	r = normalize(image[:,:,0].flatten())
	g = normalize(image[:,:,1].flatten())
	b = normalize(image[:,:,2].flatten())

	return normalize(histogram(r, -10, 10, numberInterval)), normalize(histogram(g, -10, 10, numberInterval)),\
	 	normalize(histogram(b, -10, 10, numberInterval))

def localDescriptor(image, numberRegion = 4):
	"""
	Divides the image into subregions and computes the descriptor
	"""
	height = int(image.shape[0]/numberRegion)
	width = int(image.shape[1]/numberRegion)

	localAlpha, localIntensity, localR, localG, localB = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

	for i in range(numberRegion):
		for j in range(numberRegion):
			subimage = image[i*height:(i+1)*height, j*width:(j+1)*width,:]
			alpha, intensity = hOg(subimage, numberInterval = 10)
			r, g, b = hOc(subimage, numberInterval = 10)

			localAlpha = np.concatenate((localAlpha, alpha))
			localIntensity = np.concatenate((localIntensity, intensity))
			localR = np.concatenate((localR, r))
			localG = np.concatenate((localG, g))
			localB = np.concatenate((localB, b))

	return localAlpha, localIntensity, localR, localG, localB
