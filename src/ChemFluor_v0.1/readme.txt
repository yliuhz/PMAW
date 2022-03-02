Predict File:
1. The predict molecules should be transfered from SMILES into fingerprints type at first. We recommend to use Padel to do this process.
2. Please insert solvent descriptions (five columns,Et30, SdP, SP, SA, SB) before the first column of the molecular fingerprint.
3. Delete the first line.
Example:
Name1, Et30, SP, SdP, SA, SB, CDKFingprints(1024bits),ExtFingprints(1024bits),EstateFingprints(79bits),SubFingprintsPrecence(307bits),SubFingprintsCounts(307bits)

Training File:
The first columns should be the value.
Example:
Value, Et30, SP, SdP, SA, SB, CDKFingprints(1024bits),ExtFingprints(1024bits),EstateFingprints(79bits),SubFingprintsPrecence(307bits),SubFingprintsCounts(307bits)


Put your predict file(.csv) in "put_your_predict_file_here" folder, 
and your additional training file(if you have one) in "put_your_train_file_here" folder.
Model folder contains default model and training file, editing is not recommended.


-----2020.07.20----v0.1-----
[1] Added QY regression model. We have noted the QY measurement method, and absolute QY can be oversampled 3 times, which can strike a balance between versatility and accuracy.