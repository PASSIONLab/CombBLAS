set title "Filtered BFS (25% permeability)"
set terminal png
set output "gnuplot_25.png"
set xrange [100:2100]
set yrange [0.1:256]
set logscale y
set logscale x
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Number of MPI Processes'
set ylabel 'Mean BFS Time (seconds, log scale)'
set xtics ('256' 256, '121' 121, '2048' 2048, '576' 576, '1024' 1024)
plot\
 "gnuplot_25.dat" every ::1 using 1:2:3:4 title '' ps 0 lc rgb '#FF0000' with errorbars,\
 "gnuplot_25.dat" every ::1 using 1:2 title 'Python/Python KDT' lc rgb '#FF0000' with lines,\
 "gnuplot_25.dat" every ::1 using 1:5:6:7 title '' ps 0 lc rgb '#8B0000' with errorbars,\
 "gnuplot_25.dat" every ::1 using 1:5 title 'Python/SEJITS KDT' lc rgb '#8B0000' with lines,\
 "gnuplot_25.dat" every ::1 using 1:8:9:10 title '' ps 0 lc rgb '#0000FF' with errorbars,\
 "gnuplot_25.dat" every ::1 using 1:8 title 'SEJITS/SEJITS KDT' lc rgb '#0000FF' with lines,\
 "gnuplot_25.dat" every ::1 using 1:11:12:13 title '' ps 0 lc rgb '#DAA520' with errorbars,\
 "gnuplot_25.dat" every ::1 using 1:11 title 'C++/C++ CombBLAS' lc rgb '#DAA520' with lines
