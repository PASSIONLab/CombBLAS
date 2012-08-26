set title "Filter 10"
set terminal png
set output "gnuplot_10.png"
set xrange [0:40]
set yrange [0.1:256]
set logscale y
set xlabel 'number of MPI processes'
set ylabel 'mean BFS time (s)'
set xtics ('1' 1, '36' 36, '9' 9, '16' 16, '25' 25, '4' 4)
plot\
 "gnuplot_10.dat" every ::1 using 1:2:3:4 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:2 title 'PythonSR_PythonFilter_OTF' with lines,\
 "gnuplot_10.dat" every ::1 using 1:5:6:7 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:5 title 'PythonSR_SejitsFilter_OTF' with lines,\
 "gnuplot_10.dat" every ::1 using 1:8:9:10 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:8 title 'C++SR_PythonFilter_OTF' with lines,\
 "gnuplot_10.dat" every ::1 using 1:11:12:13 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:11 title 'C++SR_SejitsFilter_OTF' with lines,\
 "gnuplot_10.dat" every ::1 using 1:14:15:16 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:14 title 'SejitsSR_SejitsFilter_OTF' with lines,\
 "gnuplot_10.dat" every ::1 using 1:17:18:19 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_10.dat" every ::1 using 1:17 title 'C++SR_PythonFilter_Mat' with lines
