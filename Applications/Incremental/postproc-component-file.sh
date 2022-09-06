#!/bin/bash

nlines=$(wc -l < $1)
line=$nlines" "$nlines
command="sed -i '1s/^/"$line"\n/' "$1
echo $command
#$($command)
