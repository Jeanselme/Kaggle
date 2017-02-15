import csv
import numpy as np

def read_labels(filename):
	y = []
	with open(filename) as f:
		next(f, None)
		train_reader = csv.reader(f)
		for line in train_reader:
			y.append(float(line[1]))
	return np.array(y)


def write_labels(filename, labels):
	with open(filename, 'w') as f:
		f.write("Id,Prediction\n")
		for i in range(len(labels)):
			f.write("%d,%d\n" % (i+1, int(labels[i])))
			
			
labels1 = read_labels("Yte-53.csv")
labels2 = read_labels("Yte-55.csv")
labels3 = read_labels("Yte-57.csv")

nb_items = len(labels1)
labels = np.zeros([nb_items])
for i in range(nb_items):
	v1 = labels1[i]
	v2 = labels2[i]
	v3 = labels3[i]
	
	if v1 == v2:
		labels[i] = v1
	else:
		labels[i] = v3
		
write_labels("Ymerge.csv", labels)
