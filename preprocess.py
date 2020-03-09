import os
import numpy as np
import pandas as pd
import pickle
import argparse
from glob import glob
import pyedflib
from collections import defaultdict

import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem import WordNetLemmatizer
# import re

# def preprocess(text):
#     stop_words = set(stopwords.words('english'))
#     stemmer = SnowballStemmer('english')
#     # covert text to lower case
#     text = text.lower()
#     text = re.sub(r'[^a-z]+', ' ', text)

#     # remove whitespace
#     text = text.strip()
#     # tokenize
#     token = word_tokenize(text)

#     # remove stop words
#     filtered = [i for i in token if not i in stop_words]
#     # stemming
#     stem = [stemmer.stem(i) for i in filtered]

#     return stem

def preprocess(file_path):
    """Preprocess files in file_path

    Arg:
    file_path: files path
    Return:
    [(eeg, report), ...], word_bag, unified frequency
    """
    data = []
    word_bag = defaultdict(int)
    freq = 250
    for dir in os.listdir(file_path):
        dir = os.path.join(file_path, dir)

        txt_f = glob(dir+"/*.txt")[0]
        report = parse_txt(txt_f, word_bag)

        edf_f_list = glob(dir+"/*.edf")
        eeg_raw, freq_raw = read_edf(edf_f_list)

        eeg = resize_eeg(eeg_raw, freq_raw, freq)
        # eeg = None

        data.append((eeg, report))
    return data, word_bag, freq

def parse_txt(txt_f, word_bag):
    """TO DO: parse report IMPRESSION & DESCRIPTION OF THE RECORD as string list, update word bag

    Some keywords are Summary of Findings & Description, not IMPRESSION & DESCRIPTION OF THE RECORD. We should consider that later.
    "In addition, we also process the reports by tokenizing and converting to lower-cases."  --EEGtoText: Learning to Write Medical Reports from EEG Recordings

    Arg:
    txt_f: report.txt path
    word_bag: {word: frequency} which should be updated
    Return:
    [IMPRESSION, DESCRIPTION OF THE RECORD]
    """
    f = open(txt_f, "r")
    lines = f.readlines()
    #print(lines)
    #print(type(lines))
    # TO DO:

    
    impression_prefix = 'IMPRESSION:'
    impression = ''
    impression_flag = False

    description_prefix = 'DESCRIPTION OF THE RECORD:'
    description = ''

    clinical_prefix = 'CLINICAL CORRELATION:'

    for line in lines:
    	line = line.lstrip()
    	line = line.rstrip()
    	if line.startswith(description_prefix):
    		description = line
    	elif line.startswith(impression_prefix):
    		impression_flag = True
    	elif line.startswith(clinical_prefix):
    		impression_flag = False

    	if impression_flag:
    		impression += line + ' '

    description = description[len(description_prefix):]
    impression = impression[len(impression_prefix):]

    description_token = word_tokenize(description.lower())
    impression_token = word_tokenize(impression.lower())

    for token in description_token:
    	word_bag[token] += 1

    for token in impression_token:
    	word_bag[token] += 1

    # print(description_token)
    # print(impression_token)
    # print(word_bag)
    return [impression, description]

def read_edf(edf_f_list):
    """TO DO: read EEG recording and calculate value for each channel

    Arg:
    edf_f_list: eeg.edf file list
    Return:
    np.array(18, SampleLength), sample frequency

    16+2 channels(5 chains)
    channels: ('EEG FP1-REF' - 'EEG F7-REF'), ('EEG F7-REF' - 'EEG T3-REF'), ...
    chains: 'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF'
            'EEG FP1-REF','EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF'
            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
            'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF'
            'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG O2-REF'
    """
    eeg_raw = []
    for edf_f in edf_f_list:
        f = pyedflib.EdfReader(edf_f)
        # # print("birthdate patientname technician", len(f.birthdate), f.birthdate, f.patientname, f.technician)
        # #print("getFileDuration(self)", f.getFileDuration())
        # #print("samplefrequency", f.samplefrequency(0), f.samplefrequency(28))
        # #print("samples_in_datarecord", f.samples_in_datarecord(0), f.samples_in_datarecord(28))
        print("datarecords_in_file: {}s*{}  {}".format(f.datarecord_duration, f.datarecords_in_file, f.file_duration))
        print("startdate: {}/{}/{} {}:{}:{}".format(f.startdate_month, f.startdate_day, f.startdate_year,
                                                    f.starttime_hour, f.starttime_minute, f.starttime_second))
        print("signals_in_file", f.signals_in_file)
        print("getSampleFrequencies", len(f.getSampleFrequencies()), f.getSampleFrequencies())
        print("samples_in_file", f.samples_in_file(0), f.samples_in_file(1), f.samples_in_file(27), f.samples_in_file(28), f.samples_in_file(29))
        signal_labels = f.getSignalLabels()
        label_used = ['EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF',
                                    'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                    'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF',
                                'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF']
        sigbufs = np.zeros((len(label_used), f.getNSamples()[0]))
        for i, label in enumerate(signal_labels):
            if label in label_used:
                idx = label_used.index(label)
                sigbufs[idx, :] = f.readSignal(i)
        print("sigbufs", sigbufs, "\n")
        chnbufs = np.zeros((18, f.getNSamples()[0]))
        # TO DO:

        eeg_raw.append(chnbufs)
        freq_raw = f.samplefrequency(0)
        f.close()
    eeg_raw = np.concatenate(tuple(eeg_raw), axis=1)
    return eeg_raw, freq_raw

def resize_eeg(eeg_raw, freq_raw, freq):
    """TO DO: resize eeg_raw from (18, freq_raw) per second to (18, freq) per second

    Need to find library to do linear interpolation

    Args:
    eeg_raw: np.array(18, SampleLength)
    freq_raw: frequency of sample
    freq: frequency we want to uniform to
    Return:
    np.array(18, SampleLengthUniformed)
    """
    eeg = eeg_raw
    return eeg

if __name__ == '__main__':
    file_path = "dataset/"
    data = preprocess(file_path)
    pickle.dump(data, open("./eeg_text.pkl", "wb"))
