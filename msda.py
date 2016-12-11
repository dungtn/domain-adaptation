import numpy as np
import numpy.matlib
import math, time


def mDA(data, prob_corruption=None, use_nonlinearity = True, mapping = None):
	if mapping is None:
		mapping = compute_reconstruction_mapping(data, prob_corruption)
	representation = np.dot(data, mapping) 
	if use_nonlinearity:
		representation = np.tanh(representation) 
	return mapping, representation


def compute_reconstruction_mapping(data, prob_corruption):
	if not (np.issubdtype(data.dtype, np.float) or np.issubdtype(data.dtype, np.integer)):
		print "data type ", data.dtype
		data.dtype = "float64"
	num_features = data.shape[1]

	feature_corruption_probs = np.ones((num_features, 1))*(1-prob_corruption)
	bias = False
	try:
		if np.allclose(np.ones((num_features,1)),data[:,-1]):
			bias = True
	except Exception as e:
		raise ValueError(e)
	if bias: 
		feature_corruption_probs[-1] = 1 
	scatter_matrix = np.dot(data.transpose(), data)
	Q = scatter_matrix*(np.dot(feature_corruption_probs, feature_corruption_probs.transpose()))
	Q[np.diag_indices_from(Q)] = feature_corruption_probs[:,0] * np.diag(scatter_matrix)
	P = scatter_matrix * numpy.matlib.repmat(feature_corruption_probs, 1, num_features)

	A = Q + 10**-5*np.eye(num_features)
	B = P
	mapping = np.linalg.solve(A.transpose(), B.transpose())
	return mapping


def mSDA(data, prob_corruption, num_layers, use_nonlinearity = True, precomp_mappings = None):
	num_data, num_features = data.shape
	mappings = list()
	representations = list()
	representations.append(data)

	if precomp_mappings is None:
		for layer in range(0, num_layers):
			mapping, representation = mDA(representations[-1], prob_corruption, use_nonlinearity)
			mappings.append(mapping)
			representations.append(representation)
	else:
		for layer in range(0, num_layers):
			mapping, representation = mDA(representations[-1], prob_corruption, use_nonlinearity, precomp_mappings[layer])
			representations.append(representation)
		mappings = precomp_mappings
	return mappings, np.asarray(representations)

