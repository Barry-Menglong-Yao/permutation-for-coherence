# permutation-for-coherence


Author: 
    Jayant Chhillar
    Barry Yao
    Vaibhav Pulastya
    


## objective
 

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
1, run model with real data
1.1, obtain nips dataset from https://www.kaggle.com/benhamner/nips-papers
1.2, put its "papers.csv" in path 'data/real/preprocess/papers.csv'
1.3, run it: "python main.py --parallel=none --predict_type=rank --criterion=fenchel_young"
 

## How to get the current best score (0.30):
If your GPU RAM> 12 GB:
    python main.py  --batch_size=8 --max_len=106 --max_sent_num=13
else:
    #You need 2 GPUs to do model parallel.
    python main.py --parallel=model --gpu=0 --gpu2=1 --batch_size=8 --max_len=106 --max_sent_num=13
    # --max_sent_num will only choose example with less than 13 sentences
    # --max_len will cut to only 106 tokens.

## How to do the preprocessing:
No need special preprocessing. 