#neural network recurrent network
#generate new city names
#regression problem using classification

#LSTM("long sort term memoty ") NN to generate city names
from __future inport absolute import , divison , print fucntion

import os 
from six import moves #url like module , pull data from url

import ssl
import tflearn
from tflearn.data_utils import 

#Step 1 - Retrieve the data 

path = "US_cities.txt"

if not os.path.isfile(path)
	context = ssl._create_unverified_context()
	#get data set
	moves.urllib.request.urlretrieve("https://raw.githubuser.com/tflearn/tflearn.github.io/master/resource/us_cities", path, context = context)

# city name max length
maxlen =20

#vectorize the text file
X, Y, char_idx = textfile_to_semi_redundant_sequences(path, 
	seq_maxlen = maxlen, redun_step = 3)

#create LSTM
#neural network layers
g = tflearn.input_data(shape = [None, maxlen, len(char_idx)])

g = tflearn.lstm(g, 512, return_seq = True)
g = tflearn.dropout(g, 0.5) #to prevent overfitting
g = tflearn.lstm(g, 512)
g = tflean.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx, activation = 'softmax')) #last layer
#softmax = logistic regression

#generate cities

m = tflearn.SequenceGenerator(g, dictionary = char_idx,
						seq_maxlen = maxlen,
						clip_gradients = 5.0,
						checkpoint_path = 'model_us_cities')


#training 

for i in range(40):
	seed = random_sequence_from_textfile(path, maxlen)
	m.fit(X, Y, validation_set = 0.1, batch_size = 128,
		n_epoch = 1, run_id = 'us cities')

	print("TESTING")
	print(m.generate(30, temperature = 1.2, seq_seed = seed))

	print("TESTING")
	print(m.generate(30, temperature = 1.0, seq_seed = seed))
	print("TESTING")
	print(m.generate(30, temperature = 0.5, seq_seed = seed))






























