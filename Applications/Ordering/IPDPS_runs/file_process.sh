#pass foldername as argument
search_dir=$1
for entry in "$search_dir"/*
do
  echo "$entry"
    sed -n '/SpMV time:/{p;}' $entry >> $entry.txt
done

