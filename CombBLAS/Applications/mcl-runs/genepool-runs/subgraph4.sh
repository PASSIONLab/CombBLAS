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

MAT1=/projectb/scratch/azad/Renamed_subgraph1_alignments_isolates_only_30_correct.txt
MAT2=/projectb/scratch/azad/Renamed_subgraph2_alignments_isolates_only_30_correct.txt
MAT3=/projectb/scratch/azad/Renamed_subgraph3_alignments_isolates_only_30_correct.txt
MAT4=/projectb/scratch/azad/Renamed_subgraph4_alignments_isolates_only_30_correct.txt
MAT5=/projectb/scratch/azad/Renamed_subgraph5_alignments_isolates_only_30_correct.txt

mcl $MAT4 --abc --partition-selection -te 16 -I 2 -p 0.0001 -S 1100 -R 1400 -pct 90 -warn-factor 0
