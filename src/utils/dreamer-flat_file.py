# This script unfolds DREAMER.mat dataset into a bunch of flat files
import gc
import os

import numpy as np
from scipy.io import loadmat

mat_data = loadmat("DREAMER.mat")  # loading mat

dreamer_data = mat_data["DREAMER"]  # dreamer dataset

dreamer_dict = {n: dreamer_data[n][0, 0]
                for n in dreamer_data.dtype.names
                }  # unwrapping dreamer dataset into a dict

num_subjects = dreamer_dict["noOfSubjects"][0, 0]  # number of subjects
num_videos = dreamer_dict["noOfVideoSequences"][0, 0]  # number of film clips

eeg_data = dreamer_dict["Data"]  # data containing eeg and scores

os.makedirs("dreamer", exist_ok=True)  # creating a new dir in existing dir
for subject in range(num_subjects):
    print(f"saving subject{subject+1}")
    os.makedirs(f"dreamer/subject{subject+1}/eeg_samples",
                exist_ok=True)  # dir to store eeg signal for each subject
    os.makedirs(f"dreamer/subject{subject+1}/eeg_labels",
                exist_ok=True)  # dir to store scores for each subject
    eeg_signal = eeg_data[0, subject]["EEG"][0, 0][0, 0][
        "stimuli"]  # eeg data of a subject
    valence = eeg_data[0, subject]["ScoreValence"][0, 0]  # valence score
    arousal = eeg_data[0, subject]["ScoreArousal"][0, 0]  # arousal score
    dominance = eeg_data[0, subject]["ScoreDominance"][0, 0]  # dominance score
    for video in range(num_videos):
        eeg = eeg_signal[video,
                         0]  # eeg signal of a subject for specific film clip
        np.savetxt(
            f"dreamer/subject{subject+1}/eeg_samples/eeg{video+1}.csv",
            eeg,
            delimiter=",",
        )  # saving eeg signal into a csv file

    np.savetxt(
        f"dreamer/subject{subject+1}/eeg_labels/valence.csv",
        valence,
        delimiter=",",
        fmt="%d",
    )  # saving valence score into a csv file
    np.savetxt(
        f"dreamer/subject{subject+1}/eeg_labels/arousal.csv",
        arousal,
        delimiter=",",
        fmt="%d",
    )  # saving arousal score into a csv file
    np.savetxt(
        f"dreamer/subject{subject+1}/eeg_labels/dominance.csv",
        dominance,
        delimiter=",",
        fmt="%d",
    )  # saving arousal score into a csv file

    gc.collect()  # invoking garbage collector manually
