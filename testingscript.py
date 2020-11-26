from input_output import csv_to_dict, add_tunefamily_ids
from music_representations import extract_melodies_from_corpus

tune_family_data_path = "/media/sirivasv/DATAL/MCC/DATASUBSET/MTC-ANN-2.0/metadata/MTC-ANN-tune-family-labels.csv"
tune_family_data_keys = ["filename","tunefamily"]

tune_family_id_path = "/media/sirivasv/DATAL/MCC/DATASUBSET/mtc-fs-1.0/MTC-FS-1.0/metadata/MTC-FS.csv"
tune_family_id_keys = ["filename","songid","source_id","serial_number","page","singer_id_s","date_of_recording","place_of_recording","latitude","longitude","title","firstline","textfamily_id","tunefamily_id","tunefamily","type","voice_strophe_number","voice_strophe","image_filename_s","audio_filename","variation","confidence"]

meta_dict = csv_to_dict(tune_family_data_path, tune_family_data_keys)
for m_i, m in enumerate(meta_dict):
  meta_dict[m_i]["tunefamily_separated"] = meta_dict[m_i]["tunefamily"].replace("_", " ")
meta_dict = add_tunefamily_ids(meta_dict, tune_family_id_path, tune_family_id_keys)

music_files_path = "/media/sirivasv/DATAL/MCC/DATASUBSET/MTC-ANN-2.0/krn/"
melody_dict = extract_melodies_from_corpus(music_files_path, meta_dict)