"""
    Copyright 2015, Berit Janssen.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import itertools as iter
import similarity as sim
import copy
import operator

def annotated_phrase_identity(phrase1, phrase2, annotator_keys) :
    """ function called to check whether the label for phrase1 and phrase2 is 
    the same. Returns 1 if it is, 0 if it is not.
    """
    similarity_dict = {}
    for a in annotator_keys :
        if phrase1[a]==phrase2[a] :
            similarity_dict[a]=1
        else :
            similarity_dict[a]=0
    return similarity_dict

def filter_results(result_list, threshold, greater_or_lower, 
     sim_measure):
    """ takes a list of similarity results for finding occurrences, 
    a list of the results associated with annotated occurrences 
    and a threshold above (operator.gt) or below (operator.lt) which results 
    should be filtered"""
    filtered_list = [r for r in result_list if 
     greater_or_lower(r['matches'][sim_measure][0]['similarity'],threshold) 
     and r['query_filename']!=r['match_filename']]
    return filtered_list

def prepare_position_evaluation(result_list, mel_dict, label_dict, sign):
    """ for each result, find the annotated occurrences and tag them in the 
    match melody, together with the best algorithmic matches
    sign indicates whether the default should be extremely high or low """
    annotator_keys = ('ann1', 'ann2', 'ann3')
    output_keys = ('query_filename','query_segment_id',
     'match_filename','tunefamily_id')
    position_ev = []
    for r in result_list:
        matched_phrase = next((s for s in label_dict if 
         s['filename'] == r['query_filename'] 
         and int(s['phrase_id']) == r['query_segment_id']), None)
        match_melody = next((m for m in mel_dict if 
         m['filename']==r['match_filename']), None)
        if match_melody == None:
            continue
        algkeys = r['matches'].keys()
        # initiate the melody as containing 
        # only extremely high or low values
        evaluation_melody = [{'onset':m['onset'],'phrase_id':m['phrase_id']} 
         for m in match_melody['symbols']]
        phrase_id = 0
        # get the label of the first phrase in the melody
        relevant_phrase_label = next((s for s in label_dict if 
         s['filename']==r['match_filename'] and 
         int(s['phrase_id'])==0), None)
        comparison = annotated_phrase_identity(matched_phrase, 
         relevant_phrase_label, annotator_keys)
        for alg in algkeys:
            for m in r['matches'][alg]:
                similarity = m['similarity']
                start_onset = m['match_start_onset']
                end_onset = m['match_end_onset']
                for ev in evaluation_melody:
                    if start_onset<=ev['onset']<=end_onset:
                        ev[alg] = similarity
            for ev in evaluation_melody:
                if not alg in ev:
                    ev[alg] = sign * 16000
        for ev in evaluation_melody:
            if ev['phrase_id']!=phrase_id:
                relevant_phrase_label = next((s for s in label_dict if 
                 s['filename']==r['match_filename'] and 
                 int(s['phrase_id'])==ev['phrase_id']),None)
                phrase_id = ev['phrase_id']
                comparison = annotated_phrase_identity(matched_phrase, 
                 relevant_phrase_label, annotator_keys)
            for ann in annotator_keys:
                ev[ann] = comparison[ann]
            ann_sum = ev['ann1']+ev['ann2']+ev['ann3']
            ev['majority'] = 0
            ev['all'] = 0
            if ann_sum == 3:
                ev['majority'] = 1
                ev['all'] = 1
            elif ann_sum == 2:
                ev['majority'] = 1  
        this_ev = {key:r[key] for key in output_keys}
        this_ev['position_eval'] = evaluation_melody
        position_ev.append(this_ev)
    return position_ev              

def order_errors_by_tunefamily(result_list, phrase_dict):
    """ takes a result list of errors (false positives / false negatives), 
    and the dictionary of matched phrases for normalization purposes """
    ordered_errors_dict = []
    tunefams = set([p['tunefamily_id'] for p in phrase_dict])
    for t in tunefams:
        phrases = [p for p in phrase_dict if p['tunefamily_id'] == t]
        melodies = set([p['filename'] for p in phrases])
        num_errors = len([r for r in result_list if r['id'][:12] in melodies])
        normalize = len(phrases)
        ordered_errors_dict.append({'tunefamily_id': t, 
         'percentage': num_errors / float(normalize)})
    return ordered_errors_dict

def pattern_precision_recall(filtered_results, mel_dict, sim_dict, 
     sim_measure, annotator):
    """ return precision and recall,
    as defined by David Meredith for Three-Layer measures (first level) 
    takes the full result list with positions of matches,
    a selection of results (e.g. true positives), a dictionary of the melodies 
    with original onset positions etc., 
    and for the given similarity measure and annotator
    calculates the agreement of pattern position"""
    pattern_precision_recall = []
    for filt in filtered_results:
        ann_occurrences = []
        alg_start = filt['matches'][sim_measure][0]['match_start_onset']
        alg_end = filt['matches'][sim_measure][0]['match_end_onset']
        alg_length = alg_end - alg_start
        # get the melody in which the query is matched
        mel = next((m['symbols'] for m in mel_dict 
         if m['filename'] == filt['match_filename']), None)
        if (mel == None):
            continue
        alg_match = [s for s in mel if alg_start<= s['onset'] <= alg_end]
        matched_phrase = next((s for s in sim_dict 
         if s['filename'] == filt['query_filename'] and 
         int(s['phrase_id']) == filt['query_segment_id']), None)
        ann_labels_this_melody = [s for s in sim_dict if 
         s['filename']==filt['match_filename']]
        for l in ann_labels_this_melody:
            occurrence = annotated_phrase_identity(matched_phrase,
             l, [annotator])[annotator]
            if occurrence==1:
                ann_occurrences.append(int(l['phrase_id']))
        if ann_occurrences:
            max_overlap = -1
            ann_length = 1
            for ann in ann_occurrences:
                comparisons = []
                ann_match = [s for s in mel if s['phrase_id'] == ann]
                note_overlap = sim.cardinality_score(alg_match,ann_match)
                if note_overlap > max_overlap:
                    max_overlap = note_overlap
                    ann_length = len(ann_match)
        else:
            continue
        alg_length = len(alg_match)
        if (alg_length == 0):
            alg_length = 1
        recall = max_overlap / float(ann_length)
        precision = max_overlap / float(alg_length) 
        pattern_precision_recall.append({'precision': precision, 
         'recall': recall, 'query_filename': filt['query_filename'], 
         'query_segment_id': filt['query_segment_id'], 
         'match_filename': filt['match_filename']})
    return pattern_precision_recall

def prepare_evaluation(result_dict, label_dict) :
    """ takes the result of segment matching within melodies
    and a dictionary of annotated phrase labels
    returns a list of dictionaries with the annotated occurrence (1 or 0) 
    and the calculated similarity
    """
    annotator_keys = ('ann1', 'ann2', 'ann3')
    matches_and_annotations = []
    tunefams = set([a['tunefamily_id'] for a in label_dict])
    for r in result_dict :
        matched_phrase = next((s for s in label_dict if 
         s['filename'] == r['query_filename'] 
         and int(s['phrase_id']) == r['query_segment_id']), None)
        algkeys = r['matches'].keys()
        dict_entry = {a: r['matches'][a]['similarity'] for a in algkeys}
        ann_matches_this_melody = [s for s in label_dict if 
         s['filename']==r['match_filename']]
        if not matched_phrase :
            print(r['query_filename'],r['query_segment_id'])
            continue
        for ankey in annotator_keys:
            comparisons = [] 
            for m in ann_matches_this_melody:
                comparisons.append(annotated_phrase_identity(matched_phrase,
                 m,annotator_keys)[ankey])
            best_matching_label_sim = max(comparisons)
            dict_entry[ankey] = best_matching_label_sim
        query_segment_id = "-"+str(r['query_segment_id']) + "-"
        dict_entry['id'] = (r['query_filename'] + query_segment_id + 
         r['match_filename'])
        matches_and_annotations.append(dict_entry)
    return matches_and_annotations


def return_F_score(precision, recall, beta=1.0):
    """ given a precision and recall value, and (optional) beta for the 
    weight of the two measures, returns the F-score. """
    if precision==0.0 or recall==0.0:
        return 0
    else:
        return ((1+beta*beta) * (precision * recall) / 
         (beta*beta* (precision + recall)))