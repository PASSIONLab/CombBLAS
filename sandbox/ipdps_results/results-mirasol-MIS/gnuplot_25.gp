set title "Filter 25"
set terminal png
set output "gnuplot_25.png"
set xrange [0:40]
set yrange [0.1:256]
set logscale y
set xlabel 'number of MPI processes'
set ylabel 'mean BFS time (s)'
set xtics ('1' 1, '36' 36, '9' 9, '16' 16, '25' 25, '4' 4)
plot\
 "gnuplot_25.dat" every ::1 using 1:2:3:4 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_25.dat" every ::1 using 1:2 title 'PythonSR_PythonFilter_ER_OTF_22' with lines,\
 "gnuplot_25.dat" every ::1 using 1:5:6:7 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_25.dat" every ::1 using 1:5 title 'PythonSR_SejitsFilter_ER_OTF_22' with lines,\
 "gnuplot_25.dat" every ::1 using 1:8:9:10 title '' ps 0 lc rgb 'black' with errorbars,\
 "gnuplot_25.dat" every ::1 using 1:8 title 'SejitsSR_SejitsFilter_ER_OTF_22' with lines
