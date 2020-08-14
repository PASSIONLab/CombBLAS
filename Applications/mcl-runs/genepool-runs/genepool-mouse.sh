#!/bin/bash
#
# Set SGE options:
#
## run the job in the current working directory (where qsub is called)
#$ -cwd
## specify an email address
#$ -M azad@lbl.gov
## specify when to send the email when job is (a)borted, (b)egins, or (e)nds
#$ -m abe
## specify a 24 hour runtime
#$ -l h_rt=2:00:00
## specify the memory used, with ram.c
#$ -l ram.c=10G
## Your job info goes here

mcl /projectb/scratch/azad/mouse_gene_mcl.mtx --abc -te 16 --partition-selection -I 2 -p 0.0001 -S 1100 -R 1400 -pct 90
