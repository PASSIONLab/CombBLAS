#pass foldername as argument
search_dir=$1
for entry in "$search_dir"/*
do
  echo "$entry"
    sed -n '/spmsv median:/{p;}' $entry >> $entry.txt
    sed -n '/allgather median:/{p;}' $entry >> $entry.txt
    sed -n '/all2all median:/{p;}' $entry >> $entry.txt
done

