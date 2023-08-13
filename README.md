# CCSER using Transfer Learning and Attention-based Fusion of Wav2vec2 and Prosody features (AFTL)
*Babak Nasersharif, Navid Naderi*

This repository contains the implementation of the paper titled "Cross Corpus Speech Emotion Recognition using Transfer Learning and Attention-based Fusion of Wav2vec2 and Prosody features". The paper presents a novel approach to speech emotion recognition that leverages transfer learning and attention mechanisms to fuse Wav2vec2 and prosody features effectively. This implementation aims to reproduce the results presented in the paper and provide a basis for further research in the field of speech emotion recognition.

## Pre-requisites
* Tested on python 3.9.0 and 3.8.2, but should work on other versions as well.
Install python requirements: `pip install -r requirements.txt`.
DisVoice is needed to extract Prosody features. Please refer to https://github.com/jcvasquezc/DisVoice for more information.

**We are currently in the process of actively cleaning up the code and adding support for all corpora.**

## IEMOCAP AFTL -> Source Domain
Since IEMOCAP is considered the source corpus, the first step is to run the corresponding IEMOCAP code. Execute the `aftl_w2v2b_p_iem_c2dcmha.py` script to extract the IEMOCAP features, initialize the model, and train it using the extracted features. This can be done by running the command `python aftl_w2v2b_p_iem_c2dcmha.py`.

The script will prompt you to provide the path to the IEMOCAP directory where the five sessions are located. If there are any existing *.pt files in the corresponding directories, the script will ask if you want to remove them and extract/save the features again.

## EMODB (Berlin) AFTL -> Target Domain
EmoDB is considered one of the target corpora. Therefore, we should fine-tune a model that has been trained on IEMOCAP using 20% of the EmoDB corpus, with the split based on speakers.

To do this, run the `aftl_w2v2b_p_I_emo_c2dcmha.py` script. This script will extract the EmoDB features, load the model trained on IEMOCAP, and fine-tune it using the extracted features. You can execute the script using the command `python aftl_w2v2b_p_I_emo_c2dcmha.py`.

The script will prompt you to provide the path to the EmoDB directory where all the ".wav" files are located. Since a checkpoint (saved model) is needed for fine-tuning, the script will also ask for the name of a checkpoint to load and fine-tune. If no response is given, it will load the "best_model000.pth" by default.

Finally, if there are any existing ".pt" files in the corresponding directory, the script will ask if you want to remove them and extract/save the features again.


## *TODO*
- Clear existing codes and add comments
- Add codes for other target corpora
- Add requirements and installation instructions
- Extend the work
