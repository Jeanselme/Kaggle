"""
	Saver
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import pickle

def save(structure, name):
	"""
	Saves the given structure in the file
	"""
	with open(name, 'wb') as output:
		pickle.dump(structure, output, pickle.HIGHEST_PROTOCOL)

def load(name):
	"""
	Loads a structure from the given file
	"""
	with open(name, 'rb') as input:
		return pickle.load(input)
