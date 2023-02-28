

Over all  follwing packages are needed to run this code, however I have included my envirnoment for this code in case if I am missing to include all required packages :

torch,
spacy,
torchtext,

csv,
tqdm,
numpy,
ujson,
httpx,
argparse,
GPUtil,
pydoi


after downloading , unzipping and placing glove.840B.300d into data directory
run with the script: 
python main.py  --task train/eval/preprocess/collect

link to download glove.840B.300d:http://nlp.stanford.edu/data/glove.840B.300d.zip


To run the code :
in the terminal execute the following command:

To train:

python main.py  --task train
To evaluate:

python main.py  --task  eval

To preprocess the data:

python main.py  --task  preprocess

To collect more Abstracts:
python main.py  --task  collect