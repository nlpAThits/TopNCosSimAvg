<p>
Download GloVe and Google embeddings in WOMBAT format from <a href="https://cosyne.h-its.org/nlpdl/wombat/wombat_embs_1625.zip">here</a> (GloVe) and <a href="https://cosyne.h-its.org/nlpdl/wombat/wombat_embs_1627.zip">here</a> (Google), unzip them, and put them into this directory.
</p>

<p>
The <b>original GloVe</b> embeddings (glove.840B.300d.zip) were obtained from <a href="http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip">here</a>.
We distribute the WOMBAT-imported version of the GloVe embeddings under the same license as the original: <a href="https://www.opendatacommons.org/licenses/pddl/1.0/">ODC Public Domain Dedication and Licence (PDDL)</a>.
</p>

<p>
The <b>original Google</b> embeddings (GoogleNews-vectors-negative300.bin) were obtained from <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing">here</a>.
We distribute the WOMBAT-imported version under the same license as the original: <a href="http://www.apache.org/licenses/LICENSE-2.0">Apache License 2.0</a>
<br>
<b>Important note:</b><br>
The WOMBAT version of the Google embeddings is a 929023 word <b>subset</b> of the original 3 mio words, containing the <b>single-word vocabulary items</b> only. It was created by extracting plain text word-vector pairs from the original binary file, and selecting only words without underscore characters. This was was done because multi-word expression vectors require non-trivial, vocabulary-aware tokenization, and it (massively!) reduced model size.
</p>
