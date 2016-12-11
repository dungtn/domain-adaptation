import spacy
import numpy as np
from collections import Counter

def create_input(dataset_fn, vocab_size=5000, use_embeddings=False, word_mappings=None):
    stops_file = open('data/stop_words.txt', "r") 
    stop_words = set([line.strip() for line in stops_file if line != "\n"]) 

    dataset = open(dataset_fn, 'r')

    docs, labels =  list(), list()
    for line in dataset:
        token = line.strip().split()
        doc   = [ e.split(':')[0] for e in token[:-1]]
        doc   = [ w for w in doc if w not in stop_words ]
        label = float(token[-1].split(':')[1])
        label = 1 if label >= 3 else 0
        docs.append(doc)
        labels.append(label)
    
    y = np.asarray(labels)
    
    n_samples = len(docs)
    
    if use_embeddings:
        embeddings = spacy.en.English()
        word_dim = 300
        max_doc_len = 20

        X = np.zeros((n_samples, word_dim * max_doc_len))
        for ix, doc in enumerate(docs):
        	for i, w in enumerate(doc[:max_doc_len]):
        		s = " ".join(w.split('_'))
        		emb = embeddings(s.decode('utf8'))
        		X[ix, i*word_dim:(i+1)*word_dim] = emb.vector
    else:
    	return_mappings = False

        X = np.zeros((n_samples, vocab_size))
        
        if word_mappings is None:
            return_mappings = True
            word_freqs = Counter([w for doc in docs for w in doc])
            top_words  = word_freqs.most_common(vocab_size - 1)

            ix_to_word = {k: v for k, v in enumerate(zip(*top_words)[0])}
            word_to_ix = {k: v for v, k in ix_to_word.iteritems()}
            
            word_mappings = (ix_to_word, word_to_ix)
        else:
            ix_to_word, word_to_ix = word_mappings

        for ix, doc in enumerate(docs):
            words_idx = [word_to_ix.get(w, vocab_size-1) for w in doc]
            X[ix, words_idx] = 1
        
        if return_mappings:
            return X, y, word_mappings

    return X, y

if __name__=='__main__':
	X_train, y_train, word_mappings = create_input("data/processed_stars/books/train")
	X_test, y_test = create_input("data/processed_stars/books/test", word_mappings=word_mappings)

	X_train, y_train = create_input("data/processed_stars/books/train", use_embeddings=True)
	X_test, y_test = create_input("data/processed_stars/books/test", use_embeddings=True)

