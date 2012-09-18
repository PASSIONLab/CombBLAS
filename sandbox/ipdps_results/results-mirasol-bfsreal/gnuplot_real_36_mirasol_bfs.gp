set terminal postscript eps color size 3.3,2
set output "gnuplot_real_36_mirasol_bfs.eps"
set xrange [-0.5:3.5]

set datafile missing "-"
set pointsize 1.5

set yrange [0.01:32]
set logscale y
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Twitter Input Graph'
set ylabel 'Mean BFS Time (seconds, log scale)'
set nokey
set xtics ('small' 0, 'medium' 1, 'large' 2, 'huge' 3)
plot\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($2) title 'Python/Python KDT' lt 1 lw 7 lc rgb '#FF0000' pt 5 with linespoints,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($7) title 'Python/SEJITS KDT' lt 1 lw 7 lc rgb '#228B22' pt 11 with linespoints,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($12) title 'SEJITS/SEJITS KDT' lt 1 lw 7 lc rgb '#0000FF' pt 13 with linespoints,\
 "gnuplot_real_36_mirasol_bfs.dat" every ::1 using 1:($17) title 'C++/C++ CombBLAS' lt 1 lw 7 lc rgb '#DAA520' pt 7 with linespoints
