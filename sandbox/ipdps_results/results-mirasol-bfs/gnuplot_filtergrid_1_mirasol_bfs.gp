set title "Filtered BFS (1% permeability)"
set terminal postscript eps color
set output "gnuplot_filtergrid_1_mirasol_bfs.eps"

set datafile missing "-"

set xrange [0.9:40]
set yrange [0.1:256]
set logscale y
set logscale x
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Number of MPI Processes'
set ylabel 'Mean BFS Time (seconds, log scale)'
set xtics ('1' 1, '36' 36, '9' 9, '16' 16, '25' 25, '4' 4)
plot\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:5:3:4:6 title '' ps 0 lt 1 lc rgb '#FF0000' with candlesticks,\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:($2) title 'Python/Python KDT' lc rgb '#FF0000' with lines,\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:10:8:9:11 title '' ps 0 lt 1 lc rgb '#8B0000' with candlesticks,\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:($7) title 'Python/SEJITS KDT' lc rgb '#8B0000' with lines,\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:15:13:14:16 title '' ps 0 lt 1 lc rgb '#0000FF' with candlesticks,\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:($12) title 'SEJITS/SEJITS KDT' lc rgb '#0000FF' with lines,\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:20:18:19:21 title '' ps 0 lt 1 lc rgb '#DAA520' with candlesticks,\
 "gnuplot_filtergrid_1_mirasol_bfs.dat" every ::1 using 1:($17) title 'C++/C++ CombBLAS' lc rgb '#DAA520' with lines
