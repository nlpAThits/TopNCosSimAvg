import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb

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
#    measurelist=args.measures.split(",")
    print_classifications=args.print_classifications != 'no'    
    print_evidence=args.print_evidence != 'no'
    plot_curves=args.plot_curves != 'no'
    plot_heatmaps=args.plot_heatmaps != 'no'
#    top_ns=args.top_n

    try:
        (sim_start, sim_end, sim_step)=args.sim_ts.split(":")
        # Steps for threshold values during dev, or supplied value test_ts
        ts_vals = np.arange(float(sim_start), float(sim_end)+float(sim_step), float(sim_step))
    except ValueError:
        ts_vals=[float(args.sim_ts)]

    top_ns=[]
    (n_start, n_end, n_step)=args.top_n.split(":")
    measurelist=[]
    for n in range(int(n_start), int(n_end)+int(n_step), int(n_step)):
        top_ns.append(n)

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
            elif ttype  == "both":          concept_text=conc_label_text+" "+conc_description_text                    
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
    results_for_measures= []        # For each value of measure, collect all results
    for emb in emblist:
        for units in unitlist:
            for top_n in top_ns:
                measure="top_"+str(top_n)
                # Results per param combi, will collect results for *all* values in ts_values (x axis)
                avg_plotp, avg_plotr, avg_plotf, avg_plotts, topn_plotp, topn_plotr, topn_plotf, topn_plotts = [], [], [], [], [], [], [], []    
                for ts in ts_vals:
                    if batches == 1:    # TUNING: Get first 20% tuning set from all pairs, create new list. Tuning pairs are the same in each run
                        tpairs = list(static_pairs[:int(len(static_pairs) * 0.2)]) # 20 percent
                    else:               # TESTING: Get last 80% as actual test data
                        tpairs = list(static_pairs[int(len(static_pairs) * 0.2):]) # 80 percent

                    for s in range(batches):                            
                        np.random.shuffle(tpairs)
                        c_pairs = tpairs[:int(len(tpairs) * batchsize)]
                        print("\n\nStarting batch %s of %s (%s instances)"%(str(s+1), str(batches), str(len(c_pairs))))
                        true_labels, avg_pred_labels, topn_pred_labels, mapper=[], [], [], []

                        for i, (p_id, c_id, l) in enumerate(c_pairs):
                            true_labels.append(int(l))
                            avg_pred_labels.append(0)
                            topn_pred_labels.append(0)
                            pairkey=c_id + " " + p_id
                            mapper.append(pairkey)  # For mapping c-p pair to label list index
                            
                            ###################################################################

                            avg_full_hash_key=c_id+p_id+emb+units+'avg_cos_sim'
                            try:                
                                avg_sim = RESULTCACHE[avg_full_hash_key]
                            except KeyError:    
                                avg_sim = None
                                evidence = []
                                
                            topn_full_hash_key=c_id+p_id+emb+units+'top_n'+str(top_n)
                            try:                
                                topn_sim,evidence,hm_data = RESULTCACHE[topn_full_hash_key]
                            except KeyError:    
                                topn_sim=None
                                evidence = []

                            if avg_sim == None or topn_sim == None:                            
                                tfdict1=PUBMETA[c_id+"->tf"]
                                tfdict2=PUBMETA[p_id+"->tf"]

                                tuples1,tuples2=[],[]
                                tf_weighting  = units.endswith("tokens")
                                idf_weighting = units.find("idf")!=-1

                                # Get both docs as lists of (word, vector) tuples
                                if emb == "fasttext":
                                    tuples1=fasttext_tuples(tfdict1, FASTTEXT)
                                    tuples2=fasttext_tuples(tfdict2, FASTTEXT)

                                else:   # Use wombat vectors
                                    tuples1=WB_CONN[0].get_vectors(emb, {}, for_input=[list(tfdict1.keys())], raw=False, in_order=False, ignore_oov=True, as_tuple=True, default=0.0)[0][1][0][2]
                                    tuples2=WB_CONN[0].get_vectors(emb, {}, for_input=[list(tfdict2.keys())], raw=False, in_order=False, ignore_oov=True, as_tuple=True, default=0.0)[0][1][0][2]  

                                if topn_sim == None:
                                    topn_sim, evidence, hm_data = get_top_n_cos_sim_avg(tuples1, tfdict1, tuples2, tfdict2, top_n, tf_weighting, idf_weighting, TOKEN_IDF)
                                    RESULTCACHE[topn_full_hash_key] = (topn_sim, evidence, hm_data)
                                if avg_sim == None:
                                    avg_sim = get_avg_cos_sim(tuples1, tfdict1, tuples2, tfdict2, tf_weighting, idf_weighting, TOKEN_IDF)
                                    RESULTCACHE[avg_full_hash_key]=avg_sim
                            
                            if topn_sim >= ts:  topn_pred_labels[mapper.index(pairkey)] = 1
                            if avg_sim >= ts:   avg_pred_labels[mapper.index(pairkey)] = 1

                            if print_classifications:
                                #pair_id=c_id + " " + p_id
                                tl=str(true_labels[mapper.index(pairkey)])
                                
                                avg_pl=str(avg_pred_labels[mapper.index(pairkey)])
                                if tl==avg_pl:  col = Fore.BLACK + Back.GREEN
                                else:           col = Fore.WHITE + Back.RED                                    
                                st=("AVG "+col+"True / Pred: %s / %s"+Style.RESET_ALL+" Sim: %s")%(tl, avg_pl, str(avg_sim))
                                st=st.ljust(70)
                                topn_pl=str(topn_pred_labels[mapper.index(pairkey)])
                                if tl==topn_pl:  col = Fore.BLACK + Back.GREEN
                                else:           col = Fore.WHITE + Back.RED                                    
                                st=st+(" TOP_N "+col+"True / Pred: %s / %s"+Style.RESET_ALL+" Sim: %s")%(tl, topn_pl, str(topn_sim))
                                print(st)

                            if print_evidence and evidence != []:   print(evidence)
                            if plot_heatmaps: make_heatmap(hm_data,str(i))
                                                    
                        # All instances in current batch are classified, using the current setup. Results are in pred_labels.
                        if args.mode == "dev":  # dev mode
                            # Each setup will produce one p,r, and f value, which we collect here                            
                            avg_plotp.append(metrics.precision_score(true_labels, avg_pred_labels))
                            avg_plotr.append(metrics.recall_score(true_labels, avg_pred_labels))
                            avg_plotf.append(metrics.f1_score(true_labels, avg_pred_labels))
                            avg_plotts.append(ts)

                            topn_plotp.append(metrics.precision_score(true_labels, topn_pred_labels))
                            topn_plotr.append(metrics.recall_score(true_labels, topn_pred_labels))
                            topn_plotf.append(metrics.f1_score(true_labels, topn_pred_labels))
                            topn_plotts.append(ts)

                            if print_classifications:
                                print("Batch %s evaluation:\nAVG: P: %s, R: %s, F: %s"%(str(s+1),str(avg_plotp[-1]), str(avg_plotr[-1]), str(avg_plotf[-1])))
                                print("Batch %s evaluation:\nTOP_N: P: %s, R: %s, F: %s"%(str(s+1),str(topn_plotp[-1]), str(topn_plotr[-1]), str(topn_plotf[-1])))

                        else:           # test mode
                            ps.append(metrics.precision_score(true_labels, pred_labels))
                            rs.append(metrics.recall_score(true_labels, pred_labels))
                            fs.append(metrics.f1_score(true_labels, pred_labels))
                            if print_classifications:
                                print("Batch %s evaluation:\nP: %s, R: %s, F: %s"%(str(s+1),str(ps[-1]), str(rs[-1]), str(fs[-1])))

                    # All batches using the current setup are finished, and their results are collected in plotp, plotr, plotf, and plotts.
                    print("\nTS val %s done"%"{0:.4f}".format(ts))

                # Store all batch results for current measure, include label for plot
                label = measure + " " + embname + " " + ttype + " " + units
                results_for_measures.append((measure, label, avg_plotp, avg_plotr, avg_plotf, avg_plotts, topn_plotp, topn_plotr, topn_plotf, topn_plotts))
            results_for_units.append((units, results_for_measures))
            results_for_measures=[]
        # end iteration over units
    # end iteration over emblist 

    if plot_curves:        
         # This can cause file name too long exceptions
#        make_plot(results_for_units,str(emblist)+str(unitlist)+str(measurelist)+ttype)
        make_double_plot(results_for_units,"")

    if args.mode=="test":
        print("Evaluation after %s batches:\n----------------------------"%str(batches))
        print("Embeddings:\t%s\nInput:\t\t%s\nUnits:\t\t%s\nMeasure:\t%s\nMin. Sim:\t%s\n"%(emb,ttype,units,measure,str(ts)))
        print("Mean P: %s (%s)\nMean R: %s (%s)\nMean F: %s (%s)"%(np.mean(ps),np.std(ps), np.mean(rs),np.std(rs), np.mean(fs),np.std(fs)))

def get_avg_cos_sim(tuples1, tfdict1, tuples2, tfdict2, tf_weighting, idf_weighting, TOKEN_IDF):

    avg1=tuple_average(tuples1, 
                        idf_dict=TOKEN_IDF if idf_weighting else {}, 
                        tf_dict=tfdict1 if tf_weighting else {})
    avg2=tuple_average(tuples2, 
                        idf_dict=TOKEN_IDF if idf_weighting else {}, 
                        tf_dict=tfdict2 if tf_weighting else {})
    return 1-dist.cosine(avg1,avg2)


def get_top_n_cos_sim_avg(tuples1, tfdict1, tuples2, tfdict2, top_n, tf_weighting, idf_weighting, TOKEN_IDF):

    # Convert tuples to weightedtuples of (word, vec, weight)
    weightedtuples1 = weight_tuples(tuples1, 
        tfdict=tfdict1 if tf_weighting else {}, 
        idfdict=TOKEN_IDF if idf_weighting else {})

    weightedtuples2 = weight_tuples(tuples2, 
        tfdict=tfdict2 if tf_weighting else {}, 
        idfdict=TOKEN_IDF if idf_weighting else {})

    n_weightedtuples1,n_weightedtuples2=[],[] 
    n1,n2=0,0
    o_max,i_max=50,50

    # Determine no of lines covered by top n distinct rank groups
    # Go over all tuples in weighting order and find cut-off point
    for n1 in range(len(weightedtuples1)):
        weight=weightedtuples1[n1][2]   # Get weight of current tuple
        if weight in n_weightedtuples1:
            continue # weight has been seen, so current tuple belongs in current rank group
        elif len(n_weightedtuples1)<top_n:
            n_weightedtuples1.append(weight) # we have not yet seen all n different weights
        else:
            break
    # n1 is the cut-off point in weightedtuples1
    xwords = [(word+" "+'{0:.3f}'.format(weight), weight in n_weightedtuples1) for (word,_,weight) in weightedtuples1[0:o_max]]
    
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
    ywords = [(word+" "+'{0:.3f}'.format(weight), weight in n_weightedtuples2) for (word,_,weight) in weightedtuples2[0:i_max]]

    matrix=np.zeros((i_max,o_max))        
    matches=[]

    for o_i in range(len(xwords)):      # Iterate over all tuples in wtups1 up to 2*cut-off point --> X
        o_tup = weightedtuples1[o_i]
        for i_i in range(len(ywords)):  # Iterate over all tuples in wtups2 up to 2*cut-off point --> Y
            i_tup = weightedtuples2[i_i]
            matrix[i_i][o_i]=1-dist.cosine(o_tup[1],i_tup[1])

    # Pad tuple lists
    weightedtuples1=weightedtuples1+([('dummy',0,0)]*n1)
    weightedtuples2=weightedtuples2+([('dummy',0,0)]*n2)

    all_sim_pairs=[]
    for o_i,o_tup in enumerate(weightedtuples1[0:n1]):              # Iterate over all tuples in wtups1 up to cut-off point --> X
        for i_i, i_tup in enumerate(weightedtuples2[0:n2]):         # Iterate over all tuples in wtups2 up to cut-off point --> Y
            cs=1-dist.cosine(o_tup[1],i_tup[1])                     # Compute pairwise sim as 1 - dist
            all_sim_pairs.append((o_tup[0]+" & "+i_tup[0],cs))      # Collect all sim pairs

    ####################################################        
    # Option:                                          # 
    # Use only top n pairs for averaging               #
    # Sort by pairwise sim (high to low)               #
    all_sim_pairs.sort(key=itemgetter(1), reverse=True)#
    if top_n > 1:                                      #
        all_sim_pairs=all_sim_pairs[:top_n]            # 
    ####################################################

    matches=[(s.split(" & ")[0], s.split(" & ")[1]) for (s,_) in all_sim_pairs]

    # Collect and average over all pairwise sims
    sim=np.average([c for (_,c) in all_sim_pairs])
    # Sort alphabetically by sim pair (for output purposes only)
    all_sim_pairs.sort(key=itemgetter(0))

#    RESULTCACHE[full_hash_key]=sim
    return (sim, all_sim_pairs, (xwords, ywords, matrix, matches))



def make_plot(data,outname):
    # data contains one tuple per units; create one column for each
    cols=len(data)
    # each row in data contains a tuple (units, [(measure1, plotlabel, [plist], [rlist], [flist], [tslist] 
    rows=len(data[0][1])

    fig, axes = plt.subplots(nrows = rows, ncols = cols, squeeze = False, figsize=(7 * cols, 6 * rows))
    plt.subplots_adjust(hspace = 0.5, wspace=0.25)
    global_maxf_val, global_maxf_x=0,0
    global_title=""
    for i,(unit, unitdata) in enumerate(data):
        for j,(measure,title,plotp,plotr,plotf,plotts) in enumerate(unitdata):
            maxf_val, maxf_x=0,0
            for val_index in range(len(plotf)):
                if plotf[val_index] > maxf_val:
                    maxf_val=plotf[val_index]
                    maxf_x=plotts[val_index]
                    global_maxf_val=plotf[val_index]
                    global_maxf_x=plotts[val_index]
                    global_title=title+" "+str(global_maxf_val)+" "+str(global_maxf_x)
                    print(global_title)
            xindex=j
            yindex=i
            axes[xindex, yindex].set_ylim([0.0,1.0])
            axes[xindex, yindex].plot(plotts, plotp)
            axes[xindex, yindex].plot(plotts, plotr)
            axes[xindex, yindex].plot(plotts, plotf)
            axes[xindex, yindex].axvline(x = maxf_x, ymin = 0, ymax = 1, color = 'black', linewidth = 1)
            axes[xindex, yindex].set_xlabel("Sim. Thresh. (Max. F {0:.3f}".format(maxf_val)+" @ sim {0:.3f})".format(maxf_x), fontsize=16)
            axes[xindex, yindex].set_title(title+"\n", fontsize=18)
            axes[xindex, yindex].tick_params(axis='both', which='major', labelsize=14)
            axes[xindex, yindex].legend(['P', 'R', 'F'], loc = 'upper right', fancybox = True, framealpha = 0.5, fontsize=14)
            
    fig.suptitle(global_title, fontsize=20)
    fig.savefig(PLOTPATH+outname + "_" + str(os.getpid()) + "_plot.png")


def make_double_plot(data,outname):
    # data contains one tuple per units; create one column for each
    cols=len(data)
    # each row in data contains a tuple (units, [(measure1, plotlabel, [plist], [rlist], [flist], [tslist] 
    rows=len(data[0][1])
    fig, axes = plt.subplots(nrows = rows, ncols = cols, squeeze = False, figsize=(7 * cols, 6 * rows))
    plt.subplots_adjust(hspace = 0.5, wspace=0.25)
    global_avg_maxf_val, global_avg_maxf_x, global_topn_maxf_val, global_topn_maxf_x=0,0,0,0
    global_avg_title, global_topn_title="",""

    for i,(unit, unitdata) in enumerate(data):
        for j,(measure, title, avg_plotp, avg_plotr, avg_plotf, avg_plotts, topn_plotp, topn_plotr, topn_plotf, topn_plotts) in enumerate(unitdata):
            avg_maxf_val, avg_maxf_x, topn_maxf_val, topn_maxf_x=0,0,0,0
            
            for val_index in range(len(avg_plotf)): # both should have same length
                if avg_plotf[val_index] > avg_maxf_val:
                    avg_maxf_val=avg_plotf[val_index]
                    avg_maxf_x=avg_plotts[val_index]
                if avg_plotf[val_index] > global_avg_maxf_val:
                    global_avg_maxf_val=avg_plotf[val_index]
                    global_avg_maxf_x=avg_plotts[val_index]
                    global_avg_title=title+" "+str(global_avg_maxf_val)+" "+str(global_avg_maxf_x)
                    
                if topn_plotf[val_index] > topn_maxf_val:
                    topn_maxf_val=topn_plotf[val_index]
                    topn_maxf_x=topn_plotts[val_index]
                if topn_plotf[val_index] > global_topn_maxf_val:
                    global_topn_maxf_val=topn_plotf[val_index]
                    global_topn_maxf_x=topn_plotts[val_index]
                    global_topn_title=title+" "+str(global_topn_maxf_val)+" "+str(global_topn_maxf_x)

            xindex=j
            yindex=i
            axes[xindex, yindex].set_ylim([0.0,1.0])

            for plotdata in [topn_plotp, topn_plotr, topn_plotf]:
                axes[xindex, yindex].plot(topn_plotts, plotdata, ls='solid', lw=2)
            for plotdata in [avg_plotp, avg_plotr, avg_plotf]:
                axes[xindex, yindex].plot(avg_plotts, plotdata, ls='dotted', lw=2)
                        
            fl="AVG     Max. F {0:.3f}".format(avg_maxf_val)+" @ sim {0:.3f}".format(avg_maxf_x)
            fl=fl+"\nTOP_N Max. F {0:.3f}".format(topn_maxf_val)+" @ sim {0:.3f}".format(topn_maxf_x)
            axes[xindex, yindex].set_xlabel(fl, fontsize=16)
            axes[xindex, yindex].set_title("AVG vs. "+title+"\n", fontsize=18)
            axes[xindex, yindex].tick_params(axis='both', which='major', labelsize=14)
            axes[xindex, yindex].legend(['P (top_n)', 'R (top_n)', 'F (top_n)', 'P (avg)', 'R (avg)', 'F (avg)'], loc = 'lower left', fancybox = True, framealpha = 0.5, fontsize=12)
    fig.suptitle("Best AVG: "+global_avg_title+"\nBest TOP_N: "+global_topn_title, fontsize=20)
    fig.savefig(PLOTPATH+outname + "_" + str(os.getpid()) + "_plot.png")

# Heat maps are for individual doc pairs
def make_heatmap(hm_data, outname, cmap="binary"):    
    xwords, ywords, matrix, matches=hm_data
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(30, 30), squeeze=True)
    hm=sb.heatmap(matrix, ax=axis, cmap=cmap, vmin=0, vmax=1, cbar=False, square=True)#, annot=True, fmt='3.3f', annot_kws={'size':5, 'alpha':0.0})
    hm.set_aspect('equal')
    hm.invert_yaxis()

    ticksize=14
    hm.set_xticklabels([l for (l,_) in xwords], rotation=90, **{'size':ticksize})
    hm.set_yticklabels([l for (l,_) in ywords], rotation=0, **{'size':ticksize}) 

    try:                xstart=[i for i, t in enumerate(xwords) if t[1] == False][0]
    except IndexError:  xstart=len(xwords)
    try:                ystart=[i for i, t in enumerate(ywords) if t[1] == False][0]
    except IndexError:  ystart=len(ywords)
        
    hm.hlines(ystart, 0, xstart, colors='darkgreen')
    hm.vlines(xstart, 0, ystart, colors='darkgreen')

    cellsize=11
    # Loop over data dimensions and create text annotations.
    for i in range(len(ywords)):
        for j in range(len(xwords)):
            color = 'black' if matrix[i,j] < 0.5 else 'white'
            content_props={'size':cellsize, 'color':color,'size':cellsize, 'weight':'bold'} 
            try:
                if (xwords[j][0].split(" ")[0], ywords[i][0].split(" ")[0]) in matches:
                    content_props['color']='darkgreen' if matrix[i,j] < 0.5 else 'mediumspringgreen'
                hm.text(j,i,"."+'{:.3f}'.format(matrix[i, j]).split(".")[1]+"\n", **content_props)
            except IndexError:  pass

    # Do not cut off longish xticklabels
    fig.subplots_adjust(top=0.98, bottom=0.3)
    plt.savefig(PLOTPATH+outname+ "_" + str(os.getpid()) + "_heatmap.png", bbox_inches='tight')
        
def prepro(text, PREPRO, FWORDS):
    unfolded_tf={}
    tokens=[t for t in PREPRO[0].tokenize(text, sw_symbol="") if t.lower() not in FWORDS]
    for tok in tokens:
        if tok in unfolded_tf.keys():   unfolded_tf[tok]=unfolded_tf[tok]+1
        else:                           unfolded_tf[tok]=1
    return unfolded_tf, tokens



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
#    parser.add_argument('--measures',   help='Similarity measures', choices = ['avg_cos_sim', 'top_n_cos_sim_avg'], required = True)
    parser.add_argument('--sim_ts',     help='Similarity threshold; either a single float, or a range of the form start:end:step', required = True) 
    parser.add_argument('--top_n',      help='N-value for topN_avg_cosine; either a single int, or a range of the form start:end:step', required = True)
    parser.add_argument('--print_evidence', help='Whether to output top N sim pairs when using top_n_cos_sim_avg', choices=['yes', 'no'], required = False, default='no')
    parser.add_argument('--print_classifications', help='Whether to output individual classification results', choices=['yes', 'no'], required = False, default='no')
    parser.add_argument('--plot_curves', help='Whether to plot P, R, and F curves', choices=['yes', 'no'], required = False, default='no')
    parser.add_argument('--plot_heatmaps', help='Whether to plot heat maps for each instance', choices=['yes', 'no'], required = False, default='no')

    args = parser.parse_args()
    main(args)


def sem_sim(spid1, spid2, smeasure, stringform, sembs, PUBMETA, WB_CONN, TOKEN_IDF, RESULTCACHE, FASTTEXT):

    spid1_tfdict=PUBMETA[spid1+"->tf"]
    spid2_tfdict=PUBMETA[spid2+"->tf"]

    # Try to get full sim result from cache
    # Depends on both docs, embeddings, type/token, and measure
    full_hash_key=spid1+spid2+sembs+stringform+smeasure
    # TODO: This never caches / returns the evidence yet
    try:
        return RESULTCACHE[full_hash_key],[], ()
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
        return sim, [], (), 0

    elif re.match('top_[0-9]*\_cos\_sim\_avg',smeasure):

        # Convert tuples to weightedtuples of (word, vec, weight)
        weightedtuples1 = weight_tuples(tuples1, 
            tfdict=spid1_tfdict if tf_weighting else {}, 
            idfdict=TOKEN_IDF if idf_weighting else {})

        weightedtuples2 = weight_tuples(tuples2, 
            tfdict=spid2_tfdict if tf_weighting else {}, 
            idfdict=TOKEN_IDF if idf_weighting else {})

        # Get no of distinct rank-groups to use.
        top_n=int(smeasure.split("_")[1])

        n_weightedtuples1,n_weightedtuples2=[],[] 
        n1,n2=0,0
        o_max,i_max=50,50

        # Determine no of lines covered by top n distinct rank groups
        # Go over all tuples in weighting order and find cut-off point
        for n1 in range(len(weightedtuples1)):
            weight=weightedtuples1[n1][2]   # Get weight of current tuple
            if weight in n_weightedtuples1:
                continue # weight has been seen, so current tuple belongs in current rank group
            elif len(n_weightedtuples1)<top_n:
                n_weightedtuples1.append(weight) # we have not yet seen all n different weights
            else:
                break
        # n1 is the cut-off point in weightedtuples1
        xwords = [(word+" "+'{0:.3f}'.format(weight), weight in n_weightedtuples1) for (word,_,weight) in weightedtuples1[0:o_max]]
        
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
        ywords = [(word+" "+'{0:.3f}'.format(weight), weight in n_weightedtuples2) for (word,_,weight) in weightedtuples2[0:i_max]]
	
        matrix=np.zeros((i_max,o_max))        
        matches=[]

        for o_i in range(len(xwords)):      # Iterate over all tuples in wtups1 up to 2*cut-off point --> X
            o_tup = weightedtuples1[o_i]
            for i_i in range(len(ywords)): # Iterate over all tuples in wtups2 up to 2*cut-off point --> Y
                i_tup = weightedtuples2[i_i]
                matrix[i_i][o_i]=1-dist.cosine(o_tup[1],i_tup[1])

        # Pad tuple lists
        weightedtuples1=weightedtuples1+([('dummy',0,0)]*n1)
        weightedtuples2=weightedtuples2+([('dummy',0,0)]*n2)

        all_sim_pairs=[]
        for o_i,o_tup in enumerate(weightedtuples1[0:n1]):              # Iterate over all tuples in wtups1 up to cut-off point --> X
            for i_i, i_tup in enumerate(weightedtuples2[0:n2]):         # Iterate over all tuples in wtups2 up to cut-off point --> Y
                cs=1-dist.cosine(o_tup[1],i_tup[1])                     # Compute pairwise sim as 1 - dist
                all_sim_pairs.append((o_tup[0]+" & "+i_tup[0],cs))      # Collect all sim pairs

        ####################################################        
        # Option:                                          # 
        # Use only top n pairs for averaging               #
        # Sort by pairwise sim (high to low)               #
        all_sim_pairs.sort(key=itemgetter(1), reverse=True)#
        if top_n > 1:                                      #
            all_sim_pairs=all_sim_pairs[:top_n]            # 
        ####################################################

        matches=[(s.split(" & ")[0], s.split(" & ")[1]) for (s,_) in all_sim_pairs]

        # Collect and average over all pairwise sims
        sim=np.average([c for (_,c) in all_sim_pairs])

        # Sort alphabetically by sim pair (for output purposes only)
        all_sim_pairs.sort(key=itemgetter(0))

        RESULTCACHE[full_hash_key]=sim
        return (sim, all_sim_pairs, (xwords, ywords, matrix, top_n, matches))

