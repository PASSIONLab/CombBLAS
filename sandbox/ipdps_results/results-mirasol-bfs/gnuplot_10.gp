set title "Filtered BFS (10% permeability)"
set terminal png
set output "gnuplot_10.png"
set xrange [0.9:40]
set yrange [0.1:256]
set logscale y
set logscale x
set xlabel 'Number of MPI Processes'
set ylabel 'Mean BFS Time (seconds, log scale)'
set xtics ('1' 1, '36' 36, '9' 9, '16' 16, '25' 25, '4' 4)
plot\
 "gnuplot_10.dat" every ::1 using 1:2:3:4 title '' ps 0 lc rgb '#FF0000' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:2 title 'Python/Python KDT' lc rgb '#FF0000' with lines,\
 "gnuplot_10.dat" every ::1 using 1:5:6:7 title '' ps 0 lc rgb '#8B0000' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:5 title 'Python/SEJITS KDT' lc rgb '#8B0000' with lines,\
 "gnuplot_10.dat" every ::1 using 1:8:9:10 title '' ps 0 lc rgb '#90EE90' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:8 title 'C++/Python KDT' lc rgb '#90EE90' with lines,\
 "gnuplot_10.dat" every ::1 using 1:11:12:13 title '' ps 0 lc rgb '#008000' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:11 title 'C++/SEJITS KDT' lc rgb '#008000' with lines,\
 "gnuplot_10.dat" every ::1 using 1:14:15:16 title '' ps 0 lc rgb '#0000FF' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:14 title 'SEJITS/SEJITS KDT' lc rgb '#0000FF' with lines,\
 "gnuplot_10.dat" every ::1 using 1:17:18:19 title '' ps 0 lc rgb '#FFD700' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:17 title 'C++/C++ CombBLAS' lc rgb '#FFD700' with lines,\
 "gnuplot_10.dat" every ::1 using 1:20:21:22 title '' ps 0 lc rgb '#000000' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:20 title 'C++/Python KDT (materialized)' lc rgb '#000000' with lines
