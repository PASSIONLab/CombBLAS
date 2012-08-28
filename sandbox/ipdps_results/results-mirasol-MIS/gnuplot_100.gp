set title "Filtered MIS (100% permeability)"
set terminal png
set output "gnuplot_100.png"
set xrange [0.9:40]
set yrange [0.1:256]
set logscale y
set logscale x
set xlabel 'Number of MPI Processes'
set ylabel 'Mean MIS Time (seconds, log scale)'
set xtics ('1' 1, '36' 36, '9' 9, '16' 16, '25' 25, '4' 4)
plot\
 "gnuplot_100.dat" every ::1 using 1:2:3:4 title '' ps 0 lc rgb '#FF0000' with errorbars,\
 "gnuplot_100.dat" every ::1 using 1:2 title 'Python/Python KDT' lc rgb '#FF0000' with lines,\
 "gnuplot_100.dat" every ::1 using 1:5:6:7 title '' ps 0 lc rgb '#8B0000' with errorbars,\
 "gnuplot_100.dat" every ::1 using 1:5 title 'Python/SEJITS KDT' lc rgb '#8B0000' with lines,\
 "gnuplot_100.dat" every ::1 using 1:8:9:10 title '' ps 0 lc rgb '#0000FF' with errorbars,\
 "gnuplot_100.dat" every ::1 using 1:8 title 'SEJITS/SEJITS KDT' lc rgb '#0000FF' with lines
