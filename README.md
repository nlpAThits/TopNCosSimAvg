# TopNCosSimAvg
Code for the RELATIONS 2019 Workshop paper
<br>
<b>Semantic Matching of Documents from Heterogeneous Collections: A Simple and Transparent Method for Practical Applications</b> <a href="https://arxiv.org/abs/1904.12550">arXiv</a>

# Installation
Create a new virtual environment with Python 3.6 first:
```shell
$ conda create --name topn python=3.6
$ conda activate topn
```
Clone this repository:
```shell
$ (topn) git clone https://github.com/nlpAThits/TopNCosSimAvg.git
```
Put the required files into the folders ```data```, ```wombat-data```, ```concept-project-mapping-dataset```, and ```fastText``` (see the respective README.md files in these folders).

The code in this repository uses the <a href="https://github.com/nlpAThits/WOMBAT">WOMBAT-API</a>, which can be installed as follows:
```shell
$ (topn) git clone https://github.com/nlpAThits/WOMBAT.git
$ (topn) cd WOMBAT
$ (topn) pip install .
```

Finally, install the following libraries:
```shell
$ (topn) conda install scipy scikit-learn gensim matplotlib colorama tqdm nltk==3.2.5
```

# Tuning: AVG_COS_SIM

For the AVG_COS_SIM measure, tuning comprises a brute-force search for the optimal value for the sim_ts parameter (the minimum cosine similarity). 
The start, end, and step values for sim_ts can be supplied like this:
<br>
```--sim_ts start:end:step```. 
The following call will search the whole range for 'label' for all four unit types, where <br>
```types      = -tf -idf``` <br>
```tokens     = +tf -idf``` <br>
```idf_types  = -tf +idf``` <br>
```idf_tokens = +tf +idf``` <br>

```shell
$ (topn) python perform-c-p-matching.py --input label --embeddings google --measures avg_cos_sim 
  --sim_ts 0.3:1.0:0.005 --units types,tokens,idf_types,idf_tokens --mode dev --draw_plots
```


# Reproducing the published results
![DEV results avg_cosine](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-avg.png "DEV results avg_cosine")

<p>
The following call will reproduce the top avg_cos_sim result reached when only label information is used.

```shell
$ (topn) python perform-c-p-matching.py  --mode dev --input label      --sim_ts .430 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```

Since the top results for avg_cos_sim are all yielded with basically the same setting, just change the value for --input and --sim_ts to reproduce the other top baseline results.

```shell
$ (topn) python perform-c-p-matching.py  --mode dev --input description --sim_ts .530 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```

```shell
$ (topn) python perform-c-p-matching.py  --mode dev --input both        --sim_ts .545 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```
</p>

![DEV results top_n_cos_sim_avg](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-topn.png "DEV results top_n_cos_sim_avg")


<p>
Likewise, the following calls will reproduce the three top top_n_cos_sim_avg results:

```shell
$ (topn) python perform-c-p-matching.py  --mode dev --input label      --sim_ts .345 --units tokens 
    --embeddings google  --measures top_n_cos_sim_avg --top_n 22 --print_classifications yes
```

```shell
$ (topn) python perform-c-p-matching.py  --mode dev --input description --sim_ts .345 --units idf_tokens 
    --embeddings glove --measures top_n_cos_sim_avg --top_n 6 --print_classifications yes
```

```shell
$ (topn) python perform-c-p-matching.py  --mode dev --input both         --sim_ts .310 --units idf_tokens 
    --embeddings fasttext --measures top_n_cos_sim_avg --top_n 14 --print_classifications yes
```

</p>
