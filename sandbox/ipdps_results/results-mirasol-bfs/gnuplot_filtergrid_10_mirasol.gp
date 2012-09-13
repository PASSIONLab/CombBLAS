set title "Filtered BFS (10% permeability)"
set terminal png
set output "gnuplot_filtergrid_10_mirasol.png"

set datafile missing "-"

set xrange [0.9:64]
set yrange [0.1:256]
set logscale y
set logscale x
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Number of MPI Processes'
set ylabel 'Mean BFS Time (seconds, log scale)'
set xtics ('1' 1, '36' 36, '9' 9, '16' 16, '25' 25, '4' 4)
plot\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:5:3:4:6 title '' ps 0 lc rgb '#FF0000' with candlesticks,\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:($2) title 'Python/Python KDT' lc rgb '#FF0000' with lines,\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:10:8:9:11 title '' ps 0 lc rgb '#8B0000' with candlesticks,\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:($7) title 'Python/SEJITS KDT' lc rgb '#8B0000' with lines,\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:15:13:14:16 title '' ps 0 lc rgb '#0000FF' with candlesticks,\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:($12) title 'SEJITS/SEJITS KDT' lc rgb '#0000FF' with lines,\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:20:18:19:21 title '' ps 0 lc rgb '#DAA520' with candlesticks,\
 "gnuplot_filtergrid_10_mirasol.dat" every ::1 using 1:($17) title 'C++/C++ CombBLAS' lc rgb '#DAA520' with lines
