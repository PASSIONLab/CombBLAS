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
#$ -l h_rt=240:00:00
## specify the memory used, with ram.c
#$ -l ram.c=1000G
## Your job info goes here

MAT=/scratch/azad/Renamed_alignments_isolates_only_30_correct_copy.txt
mcl /projectb/scratch/azad/Renamed_alignments_isolates_only_30_correct_copy.txt --abc --partition-selection -te 16 -I 2 -p 0.0001 -S 1100 -R 1400 -pct 90 -warn-factor 0
