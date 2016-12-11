import numpy as np
from sklearn import svm

from prepro import *
from msda import *

num_layers = 5
noises = [0.5, 0.6, 0.7, 0.8, 0.9]

domains = ['books', 'dvd', 'electronics', 'kitchen']
X_train, y_train, train_offset = [], [], [0]
X_test, y_test, test_offset = [], [], [0]

for domain in domains:
	train_path = "data/processed_stars/{0}/train".format(domain)
	test_path  = "data/processed_stars/{0}/test".format(domain)

	X, y, word_mappings = create_input(train_path)
	X_train.append(X)
	y_train.extend(y)
	train_offset.append(train_offset[-1] + len(X))

	X, y = create_input(test_path, word_mappings=word_mappings)
	X_test.append(X)
	y_test.extend(y)
	test_offset.append(test_offset[-1] + len(X))

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

noise = noises[0]
train_mappings, train_reps = mSDA(X_train, noise, num_layers)
_, test_reps = mSDA(X_test, noise, num_layers, train_mappings)

X_train_reps = np.hstack([X_train, train_reps[-1]])
X_test_reps = np.hstack([X_test, test_reps[-1]])

for i, domain in enumerate(domains):
	b, e = train_offset[i], train_offset[i+1]
	classifier = svm.SVC().fit(X_train_reps[b:e], y_train[b:e])
	b, e = test_offset[i], test_offset[i+1]
	preds = classifier.predict(X_test_reps[b:e])
	acc = np.mean(preds == y_test[b:e])
	print "Accuracy on %s domain" % domain

	for j, domain in enumerate(domains):
		if i == j:
			continue
		b, e = test_offset[j], test_offset[j+1]
		preds = classifier.predict(X_test_reps[b:e])
		acc = np.mean(preds == y_test[b:e])
		print "Adapt to %s domain" % domain
	print ""

