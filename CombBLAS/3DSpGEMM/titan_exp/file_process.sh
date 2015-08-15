#pass foldername as argument
search_dir=$1
for entry in "$search_dir"/*
do
  echo "$entry"
  sed -n '/prow pcol/{n;p;n;p;n;p;n;p;n;p}' $entry >> $entry.txt
done



#sed -n '/prow pcol/{n;p;p;p;p;p}' spGEMMexp_G500_27_65536.o2441622
