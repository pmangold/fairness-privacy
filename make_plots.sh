#!/bin/sh

mu=1

python plot.py legend

for i in equality_opportunity equalized_odds demographic_parity accuracy_parity accuracy
do
    python plot.py celebA $i $mu &
    python plot.py folktables $i $mu &
done
