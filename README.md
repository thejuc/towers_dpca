# towers_dpca

## dpca_analysis.ipynb

This jupyter notebook walks through fitting [dPCA](https://github.com/machenslab/dPCA) to a single session of neuropixel data and plots it.
Three different dPCA fittings are carried out each with different experimental conditions: (1) choice/decision and laser/inhibition, 
(2) choice/decision only, and (3) evidence only. Still needed: choice/decision and evidence.

## dpca_run_all.py

This is a python script that will run all mice and all sessions. It will save PDFs of the raw dPCA traces and traces of the distances between trajectories.
It will also save pickle files containing the raw distances. Each distance key in these pickle files is a list of numpy arrays of 
shape (n_neurons, n_time_bins). Each list element corresponds to a session under the key 'mouse_date'. Also found in this pickle file is the
fraction of trials in which the mouse performs correctly and is engaged (under 'pcorrect' and 'pengaged'). Here, engaged is defined by when 
mice are in state 3 (this is the state in which the psychometric curve is 'normal'). Further, the main and time PCs can be accessed in these 
pickle files and the names will depend on the type of dPCA run.

To use this python script to run a single .pickle file, navigate to its directory through your terminal and run:
```sh
python dpca_run_all.py [FILEPATH]
```
Where FilePath is the path to the desired pickle file. Currently, this will save the output files to '/jukebox/witten/yousuf/rotation/'. To change this,
edit the "path" variable in the python script. There must not be another folder in the save directory under the same name as pickle file or the script will fail to prevent overwriting.

## dpca_loop.sh and dpca_batch.sh

To speed up the process of running all the pickle files, these two shell scripts will allow you to send all files individually to slurm in one line
from the terminal. First, check that the path to dpca_run_all.py is correct on line 25 of dpca_batch.sh. Then in dpca_loop.sh check that lines 3, 8, 
and 11 all have the same and correct path to a directory containing only the picle files you would like to process. Line 24 should be the path the 
directory containing the dpca_run_all.py file (however, this shouldn't matter if you specific the full path to it in dpca_batch.sh). 

Finally, log in to spock and navigate to the directory containing dpca_loop.sh:
```sh
sh dpca_loop.sh
```
All files will be batched. Their folders and files will be created nearly immediately, but will be empty until the entire script finishes. You can
monitor you jobs using:
```sh
squeue -u [YOUR USERNAME]
```

## group_analysis.ipynb

Once you have run all files or at least all sessions for one file, you can move on to the group_analysis.ipynb notebook. This will load in the pickle
pickle files created and allow you to concatenate all sessions. In this notebook, the two laser off distance trajectories' averages are compared. Whichever one has a lower average is mirrored across the x-axis to make the graphs easier to interpret. This works consistently as dPCA is looking for an axis along which it can separate the two conditions, so the average of one is less than the other. Then, for each session, all four distance trajectories are concatenated and zscored across all trials and time-bins. Finally, we concatenate the trajectories across all sesssions and plot the four trajectories.

General interpretation of these plots: if the laser on distance trace decreases in value relative to the corresponding laser off distance trace, then it is moving more towards the opposite trajectory. 

Example: Below we see that the right trace with laser on distance is lower in value for the majority of the trial when compared to the right trace with laser off trace. Because these distance values are both taken relative to left trace with laser off, this suggests that the right trajectories with laser on are moving more towards the left trajectory with laser off.

![image](https://user-images.githubusercontent.com/86695930/226212229-459aceae-0f3b-487d-aeb8-30a289e02ad4.png)


