# CCSER using Transfer Learning and Attention-based Fusion of Wav2vec2 and Prosody features (AFTL)
*Babak Nasersharif, Navid Naderi*

This is the implementation for the paper "Cross Corpus Speech Emotion Recognition using Transfer Learning and Attention-based Fusion of Wav2vec2 and Prosody features".

**We are still actively cleaning and adding the codes for all corpora.**

## IEMOCAP AFTL -> Source Domain
Since IEMOCAP is considered the source corpus, we need to run the corresponding IEMOCAP code first. Run "aftl_w2v2b_p_iem_c2dcmha.py" to extract the IEMOCAP features, initialize the model, and train it using the extracted features. You can run it using: `python aftl_w2v2b_p_iem_c2dcmha.py`, it asks for the path of IEMOCAP where 5 sessions are available. Then, if there exist *.pt files in the corresponding directory, it asks if you want to remove them and extract/save again.