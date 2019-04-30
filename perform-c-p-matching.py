import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from colorama import init as cm_init
from colorama import Fore, Back, Style

import numpy as np, sys, re, subprocess, pickle, os, scipy.spatial.distance as dist, argparse
from operator import itemgetter
from wombat_api.core import connector as wb_conn
from contextlib import ExitStack
from sklearn import metrics
from numpy.random import shuffle

from gensim.models import FastText

PREPROFILE      = "./standard_scad_preprocessor.pkl"
WOMBAT_PATH     = "./wombat-data/"
IDF_PATH_TOKENS = "./data/polyglot_full.txt_unfolded.idf"
FWORDSFILE      = "./data/funcWordsPlus_cannot.txt"
CONCEPTFILE     = "./concept-project-mapping-dataset/concepts.txt"
PROJECTFILE     = "./concept-project-mapping-dataset/projects.txt"
ANNOFILE        = "./concept-project-mapping-dataset/annotations.txt"
FASTTEXTFILE    = "./fastText/cc.en.300.bin"
PLOTPATH        = "./plots/"

def main(args):

    cm_init()
    np.random.seed(4711)
    FWORDS=set()
    PREPRO,WB_CONN=[],[]
    RESULTCACHE,TOKEN_IDF,PUBMETA={},{},{}
    FASTTEXT=None

    if args.mode=="dev":    batches=1
    elif args.mode=="test": batches=10
           
    embname=args.embeddings
    unitlist=args.units.split(",")    
    ttype=args.input
    measurelist=args.measures.split(",")
    print_classifications=args.print_classifications != 'no'    
    print_evidence=args.print_evidence != 'no'
    plot_curves=args.plot_curves != 'no'

    try:
        (sim_start, sim_end, sim_step)=args.sim_ts.split(":")
        # Steps for threshold values during dev, or supplied value test_ts
        ts_vals = np.arange(float(sim_start), float(sim_end), float(sim_step))
    except ValueError:
        ts_vals=[float(args.sim_ts)]

    if "top_n_cos_sim_avg" in measurelist:
        if args.top_n == None:
            print("--top_n required for measure top_n_cos_sim_avg!")
            sys.exit()
        try:            
            (n_start, n_end, n_step)=args.top_n.split(":")
            measurelist=[]
            for n in range(int(n_start), int(n_end), int(n_step)):
                measurelist.append('top_'+str(n)+"_cos_sim_avg")
        except ValueError:
            measurelist[measurelist.index('top_n_cos_sim_avg')]='top_'+str(args.top_n)+"_cos_sim_avg"

    print("Using %s units and %s measures"%(str(len(unitlist)), str(len(measurelist))))

    if embname in ['google', 'glove']:
        WB_CONN.append(wb_conn(path=WOMBAT_PATH, create_if_missing=False))

    with open(IDF_PATH_TOKENS) as infile:
        for line in infile:
            try:                (key, val) = line.strip().split("\t")
            except ValueError:  pass
            TOKEN_IDF[key] = float(val)
        print("Read idf values for %s units from %s"%(str(len(TOKEN_IDF.keys())),IDF_PATH_TOKENS))

    if embname=="fasttext":
        print("Loading fastText model %s"%FASTTEXTFILE)
        FASTTEXT=FastText.load_fasttext_format(FASTTEXTFILE)
        print("done")
           
    with open(FWORDSFILE, "r") as fwf:
        for l in fwf: FWORDS.add(l.strip())
    with open(PREPROFILE, "rb") as ppf: 
        preprocessor = pickle.load(ppf)
    PREPRO.append(preprocessor)    

    pairs,static_pairs=[],[]
    with ExitStack() as stack:        
        # Read and align input files
        infiles = [stack.enter_context(open(i, "r")) for i in [CONCEPTFILE, PROJECTFILE, ANNOFILE]]
        for (concept,project,anno) in zip(*infiles):            
            # Concept
            full_conc_text = concept.split("\t")[1].strip()
            for q in range(len(full_conc_text)):
                # Find first upper case word position, which marks the end of the label and the start of the description
                if full_conc_text[q].lower() != full_conc_text[q]:
                    break # Upper case found

            conc_description_text   = full_conc_text[q:].replace("'"," ")
            conc_label_text         = full_conc_text[:q].replace("'"," ")

            if ttype    == "label":         concept_text=conc_label_text                 
            elif ttype  == "description":   concept_text=conc_description_text                                                                                                 
            elif ttype  == 'both':          concept_text=conc_label_text+" "+conc_description_text                    
            unfolded_tf, tokens=prepro(concept_text, PREPRO, FWORDS)
            PUBMETA[conc_label_text+" "+conc_description_text+"->tf"]=unfolded_tf

            # Project
            (proj_label, proj_title, proj_subject, proj_url, proj_content) = project.split("\t||\t")
            proj_content=proj_content.replace("CONTENT:","").strip().replace("'"," ")
            project_text=proj_content[:proj_content.find("Share your story with Science Buddies")+1].strip().replace("'"," ")
            unfolded_tf, tokens=prepro(project_text, PREPRO, FWORDS)
            PUBMETA[proj_label+"->tf"]=unfolded_tf

            # Label
            label = anno.strip()
            
            # Create labelled instance ...
            e=(proj_label, conc_label_text+" "+conc_description_text, label)
            # ... but ignore duplicates
            if e not in static_pairs: 
                static_pairs.append(e)                
    print("%s unique labelled concept-project pairs were read!"%str(len(static_pairs)))

    # Start experiments
    np.random.shuffle(static_pairs)     # Shuffle all static_pairs once
    ps, rs, fs, tuning_results = [], [], [], []
    batchsize = float(1 / batches)
        
    # Test several parameter combinations
    if      embname == "glove":     emblist = ["algo:glove;dataset:840b;dims:300;fold:0;norm:none;unit:token"]
    elif    embname == "google":    emblist = ["algo:sg;dataset:google-swes;dims:300;fold:0;norm:none;unit:token"]
    elif    embname == "fasttext":  emblist = [embname]    

    results_for_units = []          # For each value of units, collect all results
    results_for_measure = []        # For each value of measure, collect all results
    for emb in emblist:
        for units in unitlist:
            for measure in measurelist:
                plotp, plotr, plotf, plotts = [], [], [], []    # Results per param combi, will collect results for *all* values in ts_values (x axis)
                for ts in ts_vals:
                    if batches == 1:    # TUNING: Get first 20% tuning set from all pairs, create new list. Tuning pairs are the same in each run
                        tpairs = list(static_pairs[:int(len(static_pairs) * 0.2)]) # 20 percent

                    else:               # TESTING: Get last 80% as actual test data
                        tpairs = list(static_pairs[int(len(static_pairs) * 0.2):]) # 80 percent

                    for s in range(batches):                            
                        np.random.shuffle(tpairs)
                        c_pairs = tpairs[:int(len(tpairs) * batchsize)]
                        print("\n\nStarting batch %s of %s (%s instances)"%(str(s+1), str(batches), str(len(c_pairs))))
                        true_labels, pred_labels, mapper=[], [], []

                        # Create classifications for data in current batch, using current setup, incl ts. For dev, this is only done once ofr each setup.
                        for i, (p_id, c_id, l) in enumerate(c_pairs):
                            true_labels.append(int(l))
                            pred_labels.append(0)
                            mapper.append(c_id + " " + p_id)  # For mapping c-p pair to label list index
                            # For avg_cosine, evidence is an empty dummy list
                            sim, evidence = sem_sim(c_id, p_id, measure, units, emb, PUBMETA, WB_CONN, TOKEN_IDF, RESULTCACHE, FASTTEXT)
                            if sim >= ts:
                                pred_labels[mapper.index(c_id + " " + p_id)] = 1

                            if print_classifications:
                                pair_id=c_id + " " + p_id
                                tl=str(true_labels[mapper.index(pair_id)])
                                pl=str(pred_labels[mapper.index(pair_id)])
                                if tl==pl:col   =Back.GREEN
                                else:       col =Back.RED                                    
                                st=col+"True / Pred: %s / %s"+Style.RESET_ALL+" Sim: %s"
                                print(st%(tl, pl ,str(sim)))
                            if print_evidence and evidence != []: 
                                print(evidence)
                                                    
                        # All instances in current batch are classified, using the current setup. Results are in pred_labels.
                        if args.mode == "dev":  # dev mode
                            # Each setup will produce one p,r, and f value, which we collect here                            
                            plotp.append(metrics.precision_score(true_labels, pred_labels))
                            plotr.append(metrics.recall_score(true_labels, pred_labels))
                            plotf.append(metrics.f1_score(true_labels, pred_labels))
                            plotts.append(ts)
                            if print_classifications:
                                print("Batch %s evaluation:\nP: %s, R: %s, F: %s"%(str(s+1),str(plotp[-1]), str(plotr[-1]), str(plotf[-1])))

                        else:           # test mode
                            ps.append(metrics.precision_score(true_labels, pred_labels))
                            rs.append(metrics.recall_score(true_labels, pred_labels))
                            fs.append(metrics.f1_score(true_labels, pred_labels))
                            if print_classifications:
                                print("Batch %s evaluation:\nP: %s, R: %s, F: %s"%(str(s+1),str(ps[-1]), str(rs[-1]), str(fs[-1])))

                    # All batches using the current setup are finished, and their results are collected in plotp, plotr, plotf, and plotts.
                    print("\nTS val %s done"%"{0:.4f}".format(ts))
                # Store all batch results for current measure, include label for plot
                label = measure + "," + ttype + "," + units + "," + embname
                results_for_measure.append((measure, label, plotp, plotr, plotf, plotts))
            results_for_units.append((units, results_for_measure))
            results_for_measure=[]
            # end iteration over measures
        # end iteration over units
    # end iteration over emblist 

    if plot_curves:
        make_plot(results_for_units)

    if args.mode=="test":
        print("Evaluation after %s batches:\n----------------------------"%str(batches))
        print("Embeddings:\t%s\nInput:\t\t%s\nUnits:\t\t%s\nMeasure:\t%s\nMin. Sim:\t%s\n"%(emb,ttype,units,measure,str(ts)))
        print("Mean P: %s (%s)\nMean R: %s (%s)\nMean F: %s (%s)"%(np.mean(ps),np.std(ps), np.mean(rs),np.std(rs), np.mean(fs),np.std(fs)))

def make_plot(data):
    # data contains one tuple per units; create one column for each
    cols=len(data)
    # each row in data contains a tuple (units, [(measure1, plotlabel, [plist], [rlist], [flist], [tslist] 
    rows=len(data[0][1])
    fig, axes = plt.subplots(nrows = rows, ncols = cols, squeeze = False, figsize=(7 * cols, 5 * rows))
    plt.subplots_adjust(hspace = 0.5)
    for i,(unit, unitdata) in enumerate(data):
        for j,(measure,title,plotp,plotr,plotf,plotts) in enumerate(unitdata):
            maxf_val, maxf_x=0,0
            for val_index in range(len(plotf)):
                if plotf[val_index] > maxf_val:
                    maxf_val=plotf[val_index]
                    maxf_x=plotts[val_index]
            xindex=j
            yindex=i

            axes[xindex, yindex].plot(plotts, plotp)
            axes[xindex, yindex].plot(plotts, plotr)
            axes[xindex, yindex].plot(plotts, plotf)
            axes[xindex, yindex].axvline(x = maxf_x, ymin = 0, ymax = 1, color = 'black', linewidth = 1)
            axes[xindex, yindex].set(xlabel = 'Min. Sim.', ylabel = 'Value', title = title + "\nMax. F of " + "{0:.3f}".format(maxf_val) + " at min. sim. " + "{0:.3f}".format(maxf_x))
            axes[xindex, yindex].legend(['P', 'R', 'F'], loc = 'upper right', fancybox = True, framealpha = 0.5)
    name = title + "_" + str(os.getpid()) + "_plot.png"
    fig.savefig(PLOTPATH+name)
        
def prepro(text, PREPRO, FWORDS):
    unfolded_tf={}
    tokens=[t for t in PREPRO[0].tokenize(text, sw_symbol="") if t.lower() not in FWORDS]
    for tok in tokens:
        if tok in unfolded_tf.keys():   unfolded_tf[tok]=unfolded_tf[tok]+1
        else:                           unfolded_tf[tok]=1
    return unfolded_tf, tokens


def sem_sim(spid1, spid2, smeasure, stringform, sembs, PUBMETA, WB_CONN, TOKEN_IDF, RESULTCACHE, FASTTEXT):

    spid1_tfdict=PUBMETA[spid1+"->tf"]
    spid2_tfdict=PUBMETA[spid2+"->tf"]

    # Try to get full sim result from cache
    # Depends on both docs, embeddings, type/token, and measure
    full_hash_key=spid1+spid2+sembs+stringform+smeasure
    # TODO: This never caches / returns the evidence yet
    try:
        return RESULTCACHE[full_hash_key],[]
    except KeyError:
        pass

    tuples1,tuples2=[],[]
    tf_weighting  = stringform.endswith("tokens")
    idf_weighting = stringform.find("idf")!=-1
    sim=0.0

    # Get both docs as lists of (word, vector) tuples
    if sembs == "fasttext":
        tuples1=fasttext_tuples(spid1_tfdict, FASTTEXT)
        tuples2=fasttext_tuples(spid2_tfdict, FASTTEXT)

    else:   # Use wombat vectors
        tuples1=WB_CONN[0].get_vectors(sembs, {}, for_input=[list(spid1_tfdict.keys())], raw=False, in_order=False, ignore_oov=True, as_tuple=True, default=0.0)[0][1][0][2]
        tuples2=WB_CONN[0].get_vectors(sembs, {}, for_input=[list(spid2_tfdict.keys())], raw=False, in_order=False, ignore_oov=True, as_tuple=True, default=0.0)[0][1][0][2]  

    if smeasure=="avg_cos_sim":        
        avg1=tuple_average(tuples1, 
                            idf_dict=TOKEN_IDF if idf_weighting else {}, 
                            tf_dict=spid1_tfdict if tf_weighting else {})
        avg2=tuple_average(tuples2, 
                            idf_dict=TOKEN_IDF if idf_weighting else {}, 
                            tf_dict=spid2_tfdict if tf_weighting else {})
        sim=1-dist.cosine(avg1,avg2)
        RESULTCACHE[full_hash_key]=sim
        return sim, []

    elif re.match('top_[0-9]*\_cos\_sim\_avg',smeasure):

        # Convert tuples to weightedtuples of (word, vec, weight)
        weightedtuples1 = weight_tuples(tuples1, 
            tfdict=spid1_tfdict if tf_weighting else {}, idfdict=TOKEN_IDF if idf_weighting else {})
        weightedtuples2 = weight_tuples(tuples2, 
            tfdict=spid2_tfdict if tf_weighting else {}, idfdict=TOKEN_IDF if idf_weighting else {})

        # Get no of distinct rank-groups to use
        top_n=int(smeasure.split("_")[1])

        n_weightedtuples1,n_weightedtuples2=[],[]
        n1,n2=0,0
        # Determine no of lines covered by top n distinct rank groups
        # Go over all tuples in weighting order and find cut-off point
        for n1 in range(len(weightedtuples1)):
            weight=weightedtuples1[n1][2]   # Get weight of current tuple
            if weight in n_weightedtuples1:
                continue # weight has been seen
            elif len(n_weightedtuples1)<top_n:
                n_weightedtuples1.append(weight) # we have not yet seen all n different weights
            else:
                break
        # n1 is the cut-off point in weightedtuples1
        
        # Repeat for 2
        for n2 in range(len(weightedtuples2)):
            weight=weightedtuples2[n2][2]   # Get weight of current tuple
            if weight in n_weightedtuples2:
                continue # weight has been seen
            elif len(n_weightedtuples2)<top_n:
                n_weightedtuples2.append(weight) # we have not yet seen all n different weights
            else:
                break
        # n2 is the cut-off point in weightedtuples2

        # Pad tuple lists
        weightedtuples1=weightedtuples1+([('dummy',0,0)]*n1)
        weightedtuples2=weightedtuples2+([('dummy',0,0)]*n2)

        all_sim_pairs=[]
        for o_tup in weightedtuples1[0:n1]:             # Iterate over all tuples in wtups1 up to cut-off point
            for i_tup in weightedtuples2[0:n2]:         # Iterate over all tuples in wtups2 up to cut-off point
                cs=1-dist.cosine(o_tup[1],i_tup[1])     # Compute pairwise sim as 1 - dist
                all_sim_pairs.append((o_tup[0]+" & "+i_tup[0],cs))   # Collect all sim pairs

        ####################################################        
        # Option:                                          # 
        # Use only top n pairs for averaging               #
        # Sort by pairwise sim (high to low)               #
        all_sim_pairs.sort(key=itemgetter(1), reverse=True)#
        if top_n > 1:                                      #
            all_sim_pairs=all_sim_pairs[:top_n]            # 
        ####################################################

        # Collect all pairwise sims
        sim=np.average([c for (_,c) in all_sim_pairs])
        all_sim_pairs.sort(key=itemgetter(0))

        RESULTCACHE[full_hash_key]=sim
        return (sim, all_sim_pairs)        

def weight_tuples(tuplelist, tfdict={}, idfdict={}):
    wtuples=[]
    for t in tuplelist:
        try:                tf=tfdict[t[0]]
        except KeyError:    tf=1.0
        try:                idf=idfdict[t[0]]
        except KeyError:    idf=1.0
        wtuples.append((t[0],t[1],tf*idf))
    wtuples.sort(key=itemgetter(2), reverse=True)
    return wtuples



def fasttext_tuples(tfdict, ft_model, verbose=False):
    tuples=[]
    for i in tfdict.keys():
        try:                
            tuples.append((i, ft_model.wv[i]))        
        except KeyError:
            if verbose: print("No fastText vector for %"%i)
            pass
    return tuples

def tuple_average(tuplelist, tf_dict={}, idf_dict={}):
    vecs=[]
    for t in tuplelist:
        try:
            tfw=tf_dict[t[0]]
        except KeyError:
            tfw=1
        try:
            idfw=idf_dict[t[0]]
        except KeyError:
            idfw=1.0
        for u in range(tfw):
            # Append vector tfw times
            vecs.append(t[1]*idfw)
    return np.nanmean(vecs, axis=0)    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       help='dev or test', choices=['dev','test'], required = True)
    parser.add_argument('--embeddings', help='Embeddings source', choices=['glove', 'google', 'fasttext'], required = True)
    parser.add_argument('--input',      help='Input data', choices=['label', 'description', 'both'], required = True)
    parser.add_argument('--units',      help='Comma-separated list of input units (no spaces in between!). Choose any from: types,idf_types,tokens,idf_tokens', required = True)
    parser.add_argument('--measures',   help='Similarity measures', choices = ['avg_cos_sim', 'top_n_cos_sim_avg'], required = True)
    parser.add_argument('--sim_ts',     help='Similarity threshold; either a single float, or a range of the form start:end:step', required = True) 
    parser.add_argument('--top_n',      help='N-value for topN_avg_cosine; either a single int, or a range of the form start:end:step', required = False)
    parser.add_argument('--print_evidence', help='Whether to output top N sim pairs when using top_n_cos_sim_avg', choices=['yes', 'no'], required = False, default='no')
    parser.add_argument('--print_classifications', help='Whether to output individual classification results', choices=['yes', 'no'], required = False, default='no')
    parser.add_argument('--plot_curves', help='Whether to plot P, R, and F curves', choices=['yes', 'no'], required = False, default='no')

    args = parser.parse_args()
    main(args)

