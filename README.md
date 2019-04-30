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
Clone this repository
```shell
$ git clone https://github.com/nlpAThits/TopNCosSimAvg.git
```

The code in this repository uses the <a href="https://github.com/nlpAThits/WOMBAT">WOMBAT-API</a>, which can be installed as follows:
```shell
$ git clone https://github.com/nlpAThits/WOMBAT.git
$ cd WOMBAT
$ pip install .
```

![DEV results avg_cosine](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-avg.png "DEV results avg_cosine")


<p>
The following call will reproduce the top avg_cos_sim result reached when only label information is used.

```shell
$ python perform-c-p-matching.py  --mode dev --input label      --sim_ts .430 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```

Since the top results for avg_cos_sim are all yielded with basically the same setting, just change the value for --input and --sim_ts to reproduce the other top baseline results.

```shell
$ python perform-c-p-matching.py  --mode dev --input description --sim_ts .530 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```

```shell
$ python perform-c-p-matching.py  --mode dev --input both        --sim_ts .545 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```

</p>

![DEV results top_n_cos_sim_avg](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-topn.png "DEV results top_n_cos_sim_avg")


<p>
Likewise, the following calls will reproduce the three top top_n_cos_sim_avg results:

```shell
$ python perform-c-p-matching.py  --mode dev --input label      --sim_ts .345 --units tokens 
    --embeddings google  --measures top_n_cos_sim_avg --top_n 22 --print_classifications yes
```

```shell
$ python perform-c-p-matching.py  --mode dev --input description --sim_ts .345 --units idf_tokens 
    --embeddings glove --measures top_n_cos_sim_avg --top_n 6 --print_classifications yes
```

```shell
$ python perform-c-p-matching.py  --mode dev --input both         --sim_ts .310 --units idf_tokens 
    --embeddings fasttext --measures top_n_cos_sim_avg --top_n 14 --print_classifications yes
```

</p>
