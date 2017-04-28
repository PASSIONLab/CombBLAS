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
#$ -l h_rt=24:00:00
## specify the memory used, with ram.c
#$ -l ram.c=100G
## Your job info goes here

mcl /projectb/scratch/azad/hep-th-mcl.mtx --abc -te 16 -I 2 -p 0.01 -S 5 -R 6 -pct 0.9 -o hepth.mcl
