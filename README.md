# permutation-for-coherence


Author: 
    Jayant Chhillar
    Vaibhav Pulastya


## objective
train model from "" 
get embeddings
use embeddings to do the coherence task

## Directory structure:
"code":
    main.py: 
        read args from cmd and start the process 
        call the trainer.py
    trainer.py:
        read data, do the training, evaluation, testing in the minibatch.
        call our model in ./model directory 
    model directory: 
        the models class we use like our proposed model   
    utils:
        the common utils 
"input and output":
    data:
        input data and preprocessed data
    models:
        trained model parameters. save them here.
    log:
        log 
    output:
        other output like prediction.json or plot pictures
"requirement":
    requirement.txt:
        required packages

## How to run code:
1, run model with example preprocessed data which is saved in data/example

2, run model with real preprocessed data which you need to generate 
## Dataset:
    The dataset can be downloaded from 

## How to do the preprocessing: