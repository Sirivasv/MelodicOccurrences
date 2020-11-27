from input_output import csv_to_dict, add_tunefamily_ids
from music_representations import extract_melodies_from_corpus, filter_phrases
from evaluate import prepare_evaluation
from find_matches import distance_measures
import json
import time
import pickle

tune_family_data_path = "/media/sirivasv/DATAL/MCC/DATASUBSET/MTC-ANN-2.0/metadata/MTC-ANN-tune-family-labels.csv"
tune_family_data_keys = ["filename","tunefamily"]

tune_family_id_path = "/media/sirivasv/DATAL/MCC/DATASUBSET/mtc-fs-1.0/MTC-FS-1.0/metadata/MTC-FS.csv"
tune_family_id_keys = ["filename","songid","source_id","serial_number","page","singer_id_s","date_of_recording","place_of_recording","latitude","longitude","title","firstline","textfamily_id","tunefamily_id","tunefamily","type","voice_strophe_number","voice_strophe","image_filename_s","audio_filename","variation","confidence"]

music_files_path = "/media/sirivasv/DATAL/MCC/DATASUBSET/MTC-ANN-2.0/krn/"

meta_dict = csv_to_dict(tune_family_data_path, tune_family_data_keys)
for m_i, m in enumerate(meta_dict):
  meta_dict[m_i]["tunefamily_separated"] = meta_dict[m_i]["tunefamily"].replace("_", " ")
meta_dict = add_tunefamily_ids(meta_dict, tune_family_id_path, tune_family_id_keys)
ts = time.time()
with open('./MetaDict_{0}.json'.format(str(ts)), 'w') as outfile:
  json.dump(meta_dict, outfile)

melody_dict = extract_melodies_from_corpus(music_files_path, meta_dict)
ts = time.time()
pickle.dump(melody_dict, open('./MelodyDict_{0}.pkl'.format(str(ts)), "wb"))

phrase_dict = filter_phrases(melody_dict)
ts = time.time()
pickle.dump(phrase_dict, open('./PhraseDict_{0}.pkl'.format(str(ts)), "wb"))

full_results = distance_measures(melody_dict, phrase_dict, music_representation='pitch', return_positions=True, scaling=None)
ts = time.time()
pickle.dump(full_results, open('./DistanceMeasures_{0}.pkl'.format(str(ts)), "wb"))
