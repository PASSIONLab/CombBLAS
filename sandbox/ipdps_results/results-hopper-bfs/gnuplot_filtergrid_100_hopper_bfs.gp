set title "Filtered BFS (100% permeability)"
set terminal postscript eps color size 2.5,2
set output "gnuplot_filtergrid_100_hopper_bfs.eps"

set datafile missing "-"

set xrange [100:2500]
set yrange [0.1:256]
set logscale y
set logscale x
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Number of MPI Processes'
set ylabel 'Mean BFS Time (seconds, log scale)'
set xtics ('256' 256, '121' 121, '2048' 2048, '576' 576, '1024' 1024)
plot\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:5:3:4:6 title '' ps 0 lt 1 lc rgb '#FF0000' with candlesticks,\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:($2) title 'Python/Python KDT' lw 7 lc rgb '#FF0000' with lines,\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:10:8:9:11 title '' ps 0 lt 1 lc rgb '#8B0000' with candlesticks,\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:($7) title 'Python/SEJITS KDT' lw 7 lc rgb '#8B0000' with lines,\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:15:13:14:16 title '' ps 0 lt 1 lc rgb '#0000FF' with candlesticks,\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:($12) title 'SEJITS/SEJITS KDT' lw 7 lc rgb '#0000FF' with lines,\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:20:18:19:21 title '' ps 0 lt 1 lc rgb '#DAA520' with candlesticks,\
 "gnuplot_filtergrid_100_hopper_bfs.dat" every ::1 using 1:($17) title 'C++/C++ CombBLAS' lw 7 lc rgb '#DAA520' with lines
