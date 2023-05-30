#!/bin/sh

for i in equality_opportunity equalized_odds demographic_parity accuracy_parity accuracy
do
    python expe_fairness_fct_epsilon.py celebA $i &
    python expe_fairness_fct_epsilon.py folktables $i &
done

for i in equality_opportunity equalized_odds demographic_parity accuracy_parity accuracy
do
    python expe_fairness_fct_n.py celebA $i &
    python expe_fairness_fct_n.py folktables $i &
done
