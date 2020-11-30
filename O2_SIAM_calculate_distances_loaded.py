from input_output import csv_to_dict, add_tunefamily_ids
from music_representations import extract_melodies_from_corpus, filter_phrases
from evaluate import prepare_evaluation
from find_matches import distance_measures, matches_in_corpus, SIAM
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

# print("++++++++++")
# print("**********")

# phrase_dict_LOADED = {}
# phrase_dict_LOADED = pickle.load(open("PhraseDict_1606493916.2988284.pkl", "rb"))

phrase_dict_LOADED = {}
phrase_dict_LOADED = pickle.load(open("phrases.pkl", "rb"))

# print(phrase_dict_LOADED)

# ev_results = prepare_evaluation(full_results, phrase_dict)
# print(pattern_precision_recall(ev_results, melody_dict, phrase_dict, "ed", 'ann1'))
# full_results = distance_measures(melody_dict_LOADED, phrase_dict_LOADED, music_representation='pitch', return_positions=True, scaling=None)
full_results = matches_in_corpus(melody_dict_LOADED, phrase_dict_LOADED, measure=SIAM)
ts = time.time()

pickle.dump(full_results, open('./O2_SIAM_DistanceMeasures_LOADED_{0}.pkl'.format(str(ts)), "wb"))