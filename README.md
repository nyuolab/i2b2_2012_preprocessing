# I2B2 2012 Preprocessing
This repo contains code for convert tar.gz file (downloaded from [n2c2 data portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/))
to labels_SPLIT.txt and text_SPLIT.txt, where SPLIT is in \[train, dev, test\]. This data format is compatible for
NeMo [TokenClassification Model](https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Token_Classification-BioMegatron.ipynb).

The exact steps of conversion is as follows:
1. Convert .xml file to [brat](https://brat.nlplab.org/standoff.html) format
2. Convert brat to [bio](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))/iob2 format
3. Convert bio to nemo-comptabile format

Usage
```
python i2b2_2012_preprocessing.py
```
