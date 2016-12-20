"About the application":
An application of Question-Answering(sub-field Natural Language Processing) using encoder-Decoder RNNs with LSTM-cells + Attention based mechanism(TensorFlow)
I have a trained model  on Cornell Movie--Dialogs dataset. It works as an interactive chatbot named "Garrula".The model is an "embedding Seq2Seq model" built using Google's [Tensorflow API].
It is made of LSTM cells, which have an internal cell state that changes as inputs (words in a sentence in our datawrangler/vocab20000) are fed sequentially into the model.  This cell state allows the model to consider the context 
in which an input is recieved, and the output for a given input depends partially on the inputs that came before.  Our model has 20000 input and output 
nodes (one for each word in the vocabulary) and 3 hidden layers of 768 nodes each.

--model parameters---(for training which has been set to default in )
learning_rate LEARNING_RATE                           			        |Learning rate.                         	
learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR 		        	|Learning rate decays    	    
max_gradient_norm MAX_GRADIENT_NORM                   	        		|Clip gradients          	    
STEPS_PER_CHECKPOINT STEPS_PER_CHECKPOINT      		 		            |training steps per checkpoint.	

--Neural Network Architecture--

batch_size BATCH_SIZE           		                                |Batch size during training.     
size SIZE                                          		                |Size of each model layer.              
num_layers NUM_LAYERS                                                   |Number of layers in the model.         
vocab_size VOCAB_SIZE                                                   |Vocabulary size.                       
model_type MODEL_TYPE        				                            |encoder_decoder RNN with attention mechanism.
buckets BUCKETS                                      		           	|Implement the model with buckets                                                  
max_sentence_length  MAX_SENTENCE_LENGTH  			                	|Maximum sentence length for model WITHOUT buckets.
**********************************************************************************************************************************************

"About Dataset":
It contains conversational reponses(304,713 utterances) exchanged between 9,035 characters from 617 movies.
These conversational responses mimics a Question-Answering pattern which is used to build a chatbot.

"What's happening in the code":
Garrula is building a lexicon of 20,000 most common words in the entire dataset.
...

* "Installation & dependencies required":
1.  Install Tensorflow from https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html 
    according to the system requirement(CPU version or GPU verison).
2.  Install the following dependencies:
    python 2.7
    tensorflow (tested with v0.11.0)(We used this version.)
    numpy
    CUDA (for using gpu, see TensorFlow installation page for more details)
    cuDNN(we tested with version 5.0)
    nltk (natural language toolkit for tokenized the sentences)
    tqdm (for the nice progression bars)

"Running a program":

To train the model:

1.Set mode ="train" in configuration file setModelParams.rc.
2.Delete the content of checkpoint folder so that it can do a fresh start. If you will not delete then it will pickup the latest checkpoint file and start training from it.
3.run garrula.py using command line. (python garrula.py)

To test the model:

1.Set mode="test" in configuration file setModelParams.rc.
2.copy the checkpoint_after5day_training/ to checkpoints/ (to get the result we have shown in the report or else can test on any checkpoint.) 
3.run garrula.py using command line. (python garrula.py)


* NOTE:
This model is to be trained only once from the scratch till it reaches a low learning rate. We have previously trained it for over 5 days to reach a learning rate of 0.0001.
A checkpoint of that execution is provided in folder checkpoint_after5day_training/ 
The program is created in such a way as to pick up the learning rate from the LATEST CREATED checkpoint.
We suggest not to train the model again as it will use a higher Learning rate and by creating a recent checkpoint over the one we have provided.(for Testing only)

If the user need to train the model to try it out you should either be ready to wait 5 days(with our the machine configuration.) for a desired output during testing or do the following:
1. Run the program in "Train" mode.
2. Either use the checkpoints in the checkpoints/ to test the results or 
3. Delete the checkpoints and their metadata from checkpoints/ and copy the already trained checkpoint_after5day_training/ to checkpoints/ .
4. Later can compare the results for your understanding.

Otherwise you will get variations in output as shown in our report.

* Note:
we can also visualize a TensorFlow graph and the computations happening through tensorboard API.
Steps to run the graph visualization:
1. go into tensorboard/logs/
2. and run tensorboard --logdir=.
3. open the link which got generated from step 2
4. go to GRAPHS tab.(we have attached the output in sample_results&output/)
**************************************************************************************************************************************

--Small note for the beginners who gets confused between model-parameters and Hyper-parameters:
Training of artificial neural network requires four decision steps:
1. selection of model type with algorithm for example, RNN with LSTM cells + attention mechanism
2. selection of the architecture of the network: number of hidden layer, number of neurons per hidden layer,batch_normalization and pooling layer etc.
3. assignment of training parameters: learning rate, batch size .we can can try changing the learning rate, decay.
4. learning of model parameters: Model tries to identify the value of some parameters to fit the given data. Weights and biases in NN.

Hyper-parameters/meta-parameters are the parameters which in general(Sometime it is possible, need to read Research Articles to know more about it) cannot be learned using the ML algorithm. Therefore, we assign these parameters before training the model. 
Usually, architecture and training parameters(POINTs:2,3) are called hyper-parameters.