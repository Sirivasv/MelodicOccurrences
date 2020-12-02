from input_output import csv_to_dict, add_tunefamily_ids
from music_representations import extract_melodies_from_corpus, filter_phrases
from evaluate import prepare_evaluation, pattern_precision_recall, filter_results, return_F_score, prepare_position_evaluation
from find_matches import distance_measures
import json
import time
import pickle

meta_dict_LOADED = {}
with open('./MetaDict_1606493901.686324.json') as json_file:
  meta_dict_LOADED = json.load(json_file)

# melody_dict_LOADED = {}
# melody_dict_LOADED = pickle.load(open("MelodyDict_1606493916.28312.pkl", "rb"))

melody_dict_LOADED = {}
melody_dict_LOADED = pickle.load(open("melodies.pkl", "rb"))

# print(melody_dict_LOADED)

# phrase_dict_LOADED = {}
# phrase_dict_LOADED = pickle.load(open("PhraseDict_1606493916.2988284.pkl", "rb"))

phrase_dict_LOADED = {}
phrase_dict_LOADED = pickle.load(open("phrases.pkl", "rb"))

# print("++++++++++")
# print("******PHRASEDICT****")
# print(phrase_dict_LOADED)

# ev_results = prepare_evaluation(full_results, phrase_dict)
# # print(pattern_precision_recall(ev_results, melody_dict, phrase_dict, "ed", 'ann1'))
full_results_LOADED = {}
full_results_LOADED = pickle.load(open("O2_DMS_DistanceMeasures_LOADED_1606754478.4635398.pkl", "rb"))
# print("++++++++++")
# print("******FULLRESULTSLOADED****")
# print(full_results_LOADED)

def greater_than(val_x, threshold):
  return (val_x > threshold)

def lower_than(val_x, threshold):
  return (val_x < threshold)

threshold_val = 0.5
filtered_results = filter_results(full_results_LOADED, threshold_val, greater_than, sim_measure='cd')
# print("++++++++++")
# print("****FILTEREDRESULTS******")
# print(filtered_results)

sim_phrase_true_path = "/media/sirivasv/DATAL/MCC/DATASUBSET/MTC-ANN-2.0/metadata/MTC-ANN-phrase-similarity.csv"
sim_phrase_true_keys = ["filename","phrase_id","ann1","ann2","ann3"]
sim_phrase_true_dict = csv_to_dict(sim_phrase_true_path, sim_phrase_true_keys)
# print("++++++++++")
# print("*****TRUELABELS*****")
# print(sim_phrase_true_dict)

pattern_prec_recall = pattern_precision_recall(full_results_LOADED, melody_dict_LOADED, sim_phrase_true_dict, 'cd', "ann1")
# print("++++++++++")
# print("*****PRECRECALL*****")
# print(pattern_prec_recall)
sum_prec = 0.0
sum_recall = 0.0
for pattrn in pattern_prec_recall:
  sum_prec += pattrn["precision"]
  sum_recall += pattrn["recall"]

# print("++++++++++")
# print("*****PRECRECALLLEN*****")
# print((sum_prec, sum_recall, len(pattern_prec_recall)))


sum_prec /= len(pattern_prec_recall)
sum_recall /= len(pattern_prec_recall)

print("++++++++++")
print("*****PRECRECALLMEAN*****")
print((sum_prec, sum_recall))

print("++++++++++")
print("*****F1MEAN*****")
print(return_F_score(sum_prec, sum_recall))

# print("++++++++++")
# print("*****ANNSCOREROC_*****")
ann_values_sim_dict_for_roc = prepare_position_evaluation(full_results_LOADED, melody_dict_LOADED, sim_phrase_true_dict, -1.0)
# print(ann_values_sim_dict_for_roc)

# print("++++++++++")
# print("*****ANNLACOMPSCORES*****")
ann1_labels = []
cd_labels = []
for ev in ann_values_sim_dict_for_roc:
  for annotation in ev['position_eval']:
    # # print(annotation)
    ann1_labels.append(annotation['majority'])
    cd_labels.append(annotation['cd'])
# print(ann1_labels)
# print(cd_labels)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(ann1_labels, cd_labels)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()