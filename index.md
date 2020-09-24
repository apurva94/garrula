#### Garrula
A generative chatbot is an application of [Question-Answering](https://web.stanford.edu/~jurafsky/slp3/28.pdf) ( a subfield of Natural Language Processing), implemented using [Encoder-Decoder RNNs with LSTM-cells + Attention based mechanism](https://arxiv.org/pdf/1508.04025.pdf). The model is an "embedding Seq2Seq model" built using Tensorflow's Python API (A machine learning library). The model has been trained on Cornell Movie--Dialogs dataset. The reason behind using LSTM cells is that, they have an internal cell state that changes as inputs (words in a sentence in our data wrangler/vocab20000) are fed sequentially into the model. This cell state allows the model to consider the context in which an input is received, and the output for a given input depends partially on the inputs that came before. The model has 20000 input and output nodes (one for each word in the vocabulary) and 3 hidden layers of 768 nodes each.

#### "About Dataset":
Cornell Movie--Dialogs [dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) contains conversational responses(304,713 utterances) exchanged between 9,035 characters from 617 movies. These conversational responses mimic a Question-Answering pattern which is used to build a chatbot.

#### What's happening in the code:
Garrula is building a lexicon of 20,000 most common words in the entire dataset.
...

#### Installation & dependencies required:
1.  Install [Tensorflow](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html) from  
    according to the system requirement(CPU version or GPU version).
2.  Install the following dependencies:
    python 2.7
    tensorflow (tested with v0.11.0)
    numpy(package for scientific computing with python)
    CUDA (for using GPU, see TensorFlow installation page for more details)
    cuDNN(a GPU-accelerated library of primitives for deep neural networks)
    nltk (natural language toolkit for tokenizing the sentences)
    tqdm(Loops with a visual progress-meter.)

#### Running a program:

#### To train the model:
1. Set mode ="train" in configuration file setModelParams.rc.
2. Delete the content of checkpoint folder so that it can do a fresh start. If not deleted, then the program will pick up the latest checkpoint file and start training from it.
3. run garrula.py on the command line. (python garrula.py)

#### To test the model:
1. Set mode="test" in configuration file setModelParams.rc.
2. copy the checkpoint_after5day_training/ to checkpoints/ (To get good results, otherwise can be tested from any generated CHECKPOINTS.). 
3. run garrula.py using the command line. (python garrula.py)


#### NOTE:
This model is to be trained only once from the scratch till it reaches a lower value learning rate. We have previously trained it for over 5 days to reach a learning rate of 0.0001. A checkpoint of that execution is provided in folder checkpoint_after5day_training/. The program is created in such a way that it picks up the learning rate from the LATEST CREATED CHECKPOINT. We suggest to take a backup the checkpoint in checkpoint_after5day_training/ to match the results from the one which you have trained on for less time with a higher Learning rate(for Testing only). It will help you understand how Lower value of Learning rate helps to increase the performace of the Model.

#### Procedure for Testing model performance as mentioned above:
1. Run the program in "Train" mode.
2. Either use the checkpoints in the checkpoints/ to test the results or 
3. Delete the checkpoints and their metadata from checkpoints/ and copy the already trained checkpoint_after5day_training/ to checkpoints/ .
4. Later can compare the results for your understanding.

#### Note:
We can also visualize a TensorFlow graph and the computations happening through tensorboard API. It helps in coding the TensorFlow graph incrementaly. 
Steps to run the graph visualization:
1. go into tensorboard/logs/
2. and run tensorboard --logdir=.
3. open the link which got generated from step 2
4. go to GRAPHS tab.(we have attached the output in sample_results&output/)
******************************************************************************************************

##### --IMPORTANT NOTE for the beginners who get confused between MODEL-PARAMETERS and HYPER-PARAMETERS:
Training of ARTIFICIAL-NEURAL-NETWORK (ANN) requires 4 decision steps:
1. Selection of model type with algorithm:
   for example, "RNN with LSTM cells + Attention mechanism".
2. Selection of the Architecture of the Neural Network:
   for example,Number of hidden layer, number of neurons per hidden layer,batch_normalization and pooling layer etc.
3. Assignment of Training Parameters:
   for example, learning rate, batch size, decay_rate .we can can try changing the learning rate, decay.
4. Learning of Model Parameters(Weights): 
   Statistical Model tries to identify the value of some parameters to fit the given data. Weights and biases in NN or in 
   machine mearning Algorithms.

**"Hyper-parameters/meta-parameters"** are the parameters which in general(Sometime it is possible, need to read Research Articles to know more about it) cannot be learned using the ML algorithm.
Therefore, we assign/Tweak these parameters before "training the model". 
Usually, architecture and training parameters(POINTs:2,3) are called Hyper-Parameters.
*************************************************************************************************************
--model parameters---(for training which has been set to default )
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
buckets BUCKETS                                      		           	|Implement the model with buckets                    max_sentence_length  MAX_SENTENCE_LENGTH  	  		                	 |Maximum sentence length for model WITHOUT buckets.
************************************************************************************************************************

###$ Prerequisites:
Skills in GitHub, bash and Docker are highly recommended. For git & GitHub (this)[https://try.github.io/levels/1/challenges/1] tutorial will help. Managing Docker containers will help to avoid problems with installing lots of libraries as described on (here)[https://hub.docker.com/r/festline/mlcourse_open/] on a Wiki page.
