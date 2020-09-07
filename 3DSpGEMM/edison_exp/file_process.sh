#pass foldername as argument
search_dir=$1
for entry in "$search_dir"/Rop*
do
  echo "$entry"
    sed -n '/Restriction Op computed/{n;n;n;n;n;n;n;n;p;n;p;n;p;}' $entry >> $entry.txt
done

