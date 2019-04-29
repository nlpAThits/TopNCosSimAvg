# TopNCosSimAvg
Code for the RELATIONS 2019 Workshop paper

![DEV results avg_cosine](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-avg.png "DEV results avg_cosine")


<p>
The following call will reproduce the top avg_cos_sim result reached when only label information is used.

```shell
$ python perform-c-p-matching.py  --mode dev --input ``**`label`**`` --sim_ts .430 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```

Since the top results for avg_cos_sim are all yielded with basically the same setting, just change the value for --input and --sim_ts to reproduce the other top baseline results.

```shell
$ python perform-c-p-matching.py  --mode dev --input description --sim_ts .530 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```

```shell
$ python perform-c-p-matching.py  --mode dev --input both --sim_ts .545 --units idf_tokens 
    --embeddings google  --measures avg_cos_sim  --print_classifications yes
```


</p>

![DEV results top_n_cos_sim_avg](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-topn.png "DEV results top_n_cos_sim_avg")
