#pass foldername as argument
search_dir=$1
for entry in "$search_dir"/run_*
do
  echo "$entry"
    sed -n '/matrix  nprocesses nthreads/{n;p}' $entry >> $entry.txt
done


