



# Introduction:
The code in this repository is used to extract mineral information (Specifically 'cpx') , from abstracts of the papers related to geological rocs. The Original Data for the class label has not been added to this repository, but the code can be used for text classification as it has  parts preprocessing , training  and evaluation respectively.Feel free to modefiy the code according to your need for text classification , by editing   'args_.train_file' , 'args_.dev_file'   , and args in georoc_data and args.py respectively.
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