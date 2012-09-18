set terminal postscript eps color size 2.5,2
set output "gnuplot_filtergrid_10_mirasol_mis.eps"

set datafile missing "-"
set pointsize 1.5

set xrange [0.9:40]
set yrange [0.1:256]
set logscale y
set logscale x
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Number of MPI Processes'
set ylabel 'Mean MIS Time (seconds, log scale)'
set nokey
set xtics ('1' 1, '36' 36, '9' 9, '16' 16, '25' 25, '4' 4)
plot\
 "gnuplot_filtergrid_10_mirasol_mis.dat" every ::1 using 1:($2) title 'Python/Python KDT' lt 1 lw 7 lc rgb '#FF0000' pt 5 with linespoints,\
 "gnuplot_filtergrid_10_mirasol_mis.dat" every ::1 using 1:($7) title 'Python/SEJITS KDT' lt 1 lw 7 lc rgb '#228B22' pt 11 with linespoints,\
 "gnuplot_filtergrid_10_mirasol_mis.dat" every ::1 using 1:($12) title 'SEJITS/SEJITS KDT' lt 1 lw 7 lc rgb '#0000FF' pt 13 with linespoints,\
 "gnuplot_filtergrid_10_mirasol_mis.dat" every ::1 using 1:($17) title 'C++/C++ CombBLAS' lt 1 lw 7 lc rgb '#DAA520' pt 7 with linespoints
