# Differential Privacy has Bounded Impact on Fairness in Classification

This archive contains the code related to the paper "Differential
Privacy has Bounded Impact on Fairness in Classification". It contains
everything needed to reproduce all the figures from the paper.

## Getting the Data

The following instructions describe how to download the data to
reproduce our experiments. Note that celebA takes about 20MB and
folktables about 80MB of space on the disk.

### For celebA

Go to http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and download
the "Attributes Location" dataset. To use the file, you must first
format it as required by

	cat celebA.txt | sed "s/ /,/g" | sed "s/  */,/g" > celebA_preprocessed.csv

Then, save the `celebA_preprocessed.csv` file in a `data` directory.

### For folktables

To get the folktables dataset, simply run

	python folktables_download.py


## Running the Experiments

The file `compute_bounds.py` contains the code that effectively
computes the bounds we use in the paper. The two scripts
`expe_fairness_fct_epsilon.py` and `expe_fairness_fct_n.py` run all
experiments and save the results in a `results` directory. To run all
experiments on all datasets, run

	./run_all_experiments.sh


## Making the Plots

Finally, to make the plots, simply run

	./make_all_plots.sh

Plots will be saved in a `plots` directory. Note that Table 1 may be
reproduced by running

	python table_bound.py
