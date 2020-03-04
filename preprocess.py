import os
import numpy as np
import pandas as pd
import pickle
import argparse
from glob import glob
import pyedflib

def preprocess(file_path):
    """Preprocess files in file_path
    Arg:
    file_path: files path
    Return:
    [(eeg, report), ...], word_bag
    """
    data = []
    word_bag = dict()
    for dir in os.listdir(file_path):
        dir = os.path.join(file_path, dir)

        txt_f = glob(dir+"/*.txt")[0]
        report = parse_txt(txt_f, word_bag)

        edf_f_list = glob(dir+"/*.edf")
        eeg = parse_edf(edf_f_list)

        data.append((eeg, report))
    return data, word_bag

def parse_edf(edf_f_list):
    """TO DO: parse EEG recording

    Need concatenate different edf files to a np.array instead of storing them in list

    Arg:
    edf_f_list: eeg.edf file list
    Return:
    np.array(18, SampleLength)

    16+2 channels(5 chains)
    channels: ('EEG FP1-REF' - 'EEG F7-REF'), ('EEG F7-REF' - 'EEG T3-REF'), ...
    chains: 'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF'
            'EEG FP1-REF','EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF'
            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
            'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF'
            'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG O2-REF'
    """
    eeg = []
    for edf_f in edf_f_list:
        f = pyedflib.EdfReader(edf_f)
        # # print("birthdate patientname technician", len(f.birthdate), f.birthdate, f.patientname, f.technician)
        print("datarecords_in_file: {}s*{}  {}".format(f.datarecord_duration, f.datarecords_in_file, f.file_duration))
        print("startdate: {}/{}/{} {}:{}:{}".format(f.startdate_month, f.startdate_day, f.startdate_year,
                                                    f.starttime_hour, f.starttime_minute, f.starttime_second))
        print("signals_in_file", f.signals_in_file)
        # #print("getFileDuration(self)", f.getFileDuration())
        print("getSampleFrequencies", len(f.getSampleFrequencies()), f.getSampleFrequencies())
        # #print("samplefrequency", f.samplefrequency(0), f.samplefrequency(28))
        # #print("samples_in_datarecord", f.samples_in_datarecord(0), f.samples_in_datarecord(28))
        print("samples_in_file", f.samples_in_file(0), f.samples_in_file(1), f.samples_in_file(27), f.samples_in_file(28), f.samples_in_file(29))
        print('\n')
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
        print("sigbufs", sigbufs)
        chnbufs = np.zeros((18, f.getNSamples()[0]))
        # TO DO:

        eeg.append(chnbufs)
        f.close()
    # TO DO: concatenate eeg to one np.array

    return eeg

def parse_txt(txt_f, word_bag):
    """TO DO: parse report IMPRESSION & DESCRIPTION OF THE RECORD as string list, update word bag

    Sometimes keywords are not IMPRESSION & DESCRIPTION OF THE RECORD. We should include more.
    "In addition, we also process the reports by tokenizing and converting to lower-cases."

    Arg:
    txt_f: report.txt path
    word_bag: {word: frequency}
    Return:
    [IMPRESSION, DESCRIPTION OF THE RECORD]
    """
    f = open(txt_f, "r")
    print(f.read())
    print(type(f.read()))
    # TO DO:

if __name__ == '__main__':
    file_path = "dataset/"
    data = preprocess(file_path)
    pickle.dump(data, open("./eeg_text.pkl", "wb"))
