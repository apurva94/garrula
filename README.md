An application of Question-Answering(sub-field Natural Language Processing) using techniques of Deep.
We have a trained model which works as an interactive chatbot named"garrula"
garrula on a particular dataset.For example Cornell Movie--Dialogs.
The conversational reponses(304,713 utterances) exchanged between 9,035 characters from 617 movies mimics a question-Answering pattern which is used to build a chatbot.


Garrula is building a lexicon  of 20,000 most common words in the entire dataset.
1. Install Tensorflow from https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html according to the system requirent(CPU version or GPU verison)

2. Our program requires following dependencies

python 2.7
tensorflow (tested with v0.11.0)
numpy
CUDA (for using gpu, see TensorFlow installation page for more details)
cuDNN(we tested )
nltk (natural language toolkit for tokenized the sentences)
tqdm (for the nice progression bars)


3. Running of the program:

To train the model:

1.change the mode into "train" in configuration file setModelParams.rc.
2.Delete the content of checkpoint folder so that it can do a fresh start.If you will not delete then it will pickup the latest checkpoint file.
3.python garrula.py

TO test the model:

1.change the mode into "test" in configuration file setModelParams.rc.
2.copy the checkpoint_after5day_training/ to checkpoints/ (to get the result we have shown in the paper or else can test on any checkpoint.) 
3.python garrula.py

****

PLEASE NOTE:
This model is to be trained only once from the scratch till it reaches a low learning rate. We have previously trained it for over 5 days to reach a learning rate of 0.0001.
A checkpoint of that execution is provided in folder Working_dir. 
The program is created in such a way as to pick up the learning rate from the LATEST CREATED checkpoint.
We suggest not to train the model again as it will use a high rate and create a recent checkpoint over the one we have provided.

If the user need to train the model to try it out you should either be ready to wait 5 days for a desired output during testing or do the following:
1. Sort the working_dir folder according to recently created/updated files.
2. Delete the checkpoint files and checkpoint metadata files which are newly created
3. Make sure the checkpoints previously existing are still there

Else you will get variations in output as shown in our paper



****
****
we can also visualize a TensorFlow graph and the computations happening through tensorboard API.
Steps to run the graph visualization:
1. go into tensorboard/logs/
2. and run tensorboard --logdir=.
3. open the link which got generated from step 2
4. go to GRAPHS tab.(we have attached the output in sample_results&output/)

****


The model is an "embedding Seq2Seq model" built using Google's [Tensorflow].
It is made of LSTM cells, which have an internal cell state that changes 
as inputs (words in a sentence in our vocabulary_20000) are fed sequentially into the model.  This cell state allows the model to consider the context 
in which an input is recieved, and the output for a given input depends partially on the inputs that came before.  Our model has 20000 input and output 
nodes (one for each word in the vocabulary) and 3..?? hidden layers of 768 nodes each.
""
--model parameters---(for training which has been set to default in )
learning_rate LEARNING_RATE                           			|Learning rate.                         	
learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR 			|Learning rate decays    	    
max_gradient_norm MAX_GRADIENT_NORM                   			|Clip gradients          	    
STEPS_PER_CHECKPOINT STEPS_PER_CHECKPOINT      		 		|training steps per checkpoint.	

--Neural Network Architecture--

batch_size BATCH_SIZE           		                        |Batch size during training.     
size SIZE                                          		        |Size of each model layer.              
num_layers NUM_LAYERS                                                   |Number of layers in the model.         
vocab_size VOCAB_SIZE                                                   |Vocabulary size.                       
model_type MODEL_TYPE        				                |encoder_decoder RNN with attention mechanism.
buckets BUCKETS                                      			|Implement the model with buckets                                                  
max_sentence_length  MAX_SENTENCE_LENGTH  				|Maximum sentence length for model WITHOUT buckets.
###################################################################################################################################################################################################################
