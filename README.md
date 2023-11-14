



# Introduction:
The code in this repository is used to extract mineral information (Specifically 'cpx') , from abstracts of the papers related to geological rocks. The Original Data for the class label has not been added to this repository.  The initial idea was  to test if we  can turn a manual data entry into an automated one with the help of machine learning. Sepcifically, instead of reading paper by human, we give to an ml model, which performed quiet well. 
 The pdf file,  '__presentation_georoc.pdf__'   that I prepared as part of a presentation in DH seminar in the university of Goettingen is also included here. 

## Dependencies
Over all  the follwing packages are needed to run this code, however I have included my envirnoment for this code in case if I am missing to include all required packages :
```
torch,
spacy,
torchtext
```
```
csv,
tqdm,
numpy,
ujson,
httpx,
argparse,
GPUtil,
pydoi
```


link to download glove.840B.300d:http://nlp.stanford.edu/data/glove.840B.300d.zip

## Running the code
after downloading , unzipping and placing glove.840B.300d into data directory
run with the script: 


###### To train:

```
python main.py  --task train
```
###### To evaluate:

```
python main.py  --task  eval
```

###### To preprocess the data:

```
 python main.py  --task  preprocess
 ```

###### To collect more Abstracts:
```
python main.py  --task  collect
```