set title "BFS on Twitter Data (36 processes)"
set terminal postscript eps color size 2.5,2
set output "gnuplot_real_36_mirasol_bfs.eps"
set xrange [-0.5:3.5]

set datafile missing "-"

set yrange [0.01:32]
set logscale y
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Twitter Input Graph'
set ylabel 'Mean BFS Time (seconds, log scale)'
set key right top
set xtics ('small' 0, 'medium' 1, 'large' 2, 'huge' 3)
plot\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:5:3:4:6 title '' ps 0 lt 1 lc rgb '#FF0000' with candlesticks,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($2) title 'Python/Python KDT' lw 7 lc rgb '#FF0000' with lines,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:10:8:9:11 title '' ps 0 lt 1 lc rgb '#8B0000' with candlesticks,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($7) title 'Python/SEJITS KDT' lw 7 lc rgb '#8B0000' with lines,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:15:13:14:16 title '' ps 0 lt 1 lc rgb '#0000FF' with candlesticks,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($12) title 'SEJITS/SEJITS KDT' lw 7 lc rgb '#0000FF' with lines,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:20:18:19:21 title '' ps 0 lt 1 lc rgb '#DAA520' with candlesticks,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($17) title 'C++/C++ CombBLAS' lw 7 lc rgb '#DAA520' with lines
