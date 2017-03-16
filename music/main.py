#dependencies
from  deepmusic.moduleloader import moduleloader
#predict next key
from deepmusic.keyboardcell import Keyboardcell

#encapsulate song data so we can run get_scale , get_relative_methods
import deepmusic.songstruct as music
import numpy as np #generate random numbers	
import tensorflow as tf #for flowwing

def bulid_network(self):
	#create computatoion fraph encapsulate session and the graph init
	input_din = ModuleLoader.batch_builders.get_module().get_input_dim)
	
	#note date
	with tf.name_scope('placeholder_inputs'):
		self.inputs = {
		tf.placeholder(
			tf.float32,#numerical data
			[self.batch_size, input_din] ,#how much data
			name = 'input'
			)
		}


	#tagets a88 key , binary classification problesm(piano)

	with tf.name_scope('placeholder_targets'):
		self.targets = [
			tf.placeholder(tf.init32,#0/1
			[self.batch_size]),
			name = 'target')
		]


	with tf.name_scope('placeholder_use_prev'):
		self.use_prev = [
			tf.placeholder(
				tf.bool,
				[],
				name = 'use_prev'
				)

		]



