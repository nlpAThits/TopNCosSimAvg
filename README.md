# TopNCosSimAvg
Code for the RELATIONS 2019 Workshop paper

![DEV results avg_cosine](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-avg.png "DEV results avg_cosine")


<p>
The following call will reproduce the top result reached when only label information is used.

```shell
$ python perform-c-p-matching.py  --mode dev --embeddings google --input label --units idf_tokens --measures avg_cos_sim  --sim_ts .430 --print_classifications yes
```
</p>

![DEV results top_n_cos_sim_avg](https://github.com/nlpAThits/TopNCosSimAvg/blob/master/images/dev-topn.png "DEV results top_n_cos_sim_avg")
