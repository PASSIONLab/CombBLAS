#pass foldername as argument
sed -n '/summary statistics/{n;p;}' $1 >> $1.txt

