set terminal postscript eps color size 3.3,2
set output "gnuplot_filtergrid_25_hopper_bfs.eps"

set datafile missing "-"
set pointsize 1.5

set xrange [54:2500]
set yrange [0.1:256]
set logscale y
set logscale x
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Number of MPI Processes'
set ylabel 'Mean BFS Time (seconds, log scale)'
set nokey
set xtics ('64' 64, '1024' 1024, '2025' 2025, '256' 256, '576' 576, '121' 121)
plot\
 "gnuplot_filtergrid_25_hopper_bfs.dat" every ::1 using 1:($2) title 'Python/Python KDT' lt 1 lw 7 lc rgb '#FF0000' pt 5 with linespoints,\
 "gnuplot_filtergrid_25_hopper_bfs.dat" every ::1 using 1:($7) title 'Python/SEJITS KDT' lt 1 lw 7 lc rgb '#228B22' pt 11 with linespoints,\
 "gnuplot_filtergrid_25_hopper_bfs.dat" every ::1 using 1:($12) title 'SEJITS/SEJITS KDT' lt 1 lw 7 lc rgb '#0000FF' pt 13 with linespoints,\
 "gnuplot_filtergrid_25_hopper_bfs.dat" every ::1 using 1:($17) title 'C++/C++ CombBLAS' lt 1 lw 7 lc rgb '#DAA520' pt 7 with linespoints
