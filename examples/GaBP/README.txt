This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gaussian belief propagation Matlab package

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Writen by Danny Bickson, CMU, HUJI & IBM Haifa Lab

please report bugs to: danny.bickson@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Acknowledgment: the following people contributed code/their help
for this package:
N. Sommer - TAU, and E. N. Hoch, HUJI: helped in implementing the LDLC decoder code
A. Zymnis - Stanford: Wrote the NUM simulation
NBP/LDLC encoding matrices where kindly provided by Marilynn Green, Nokia Siemens Networks Research Technology Platforms Dallas, TX.
S. Joshi - Stanford, Wrote the original sensor selection simulation.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

If you are using my code, please cite the following:
@phdthesis{bicksonThesis,
  title={{Gaussian Belief Propagation: Theory and Application}},
  author={Bickson, D.},
  year={2009},
  school={The Hebrew University of Jerusalem},
}

The following files are basic implementations of the
GaBP algorithm. The code is given for education pruposes, and is not
optimized. Only the sparse version (3) is optimized.

1) gabp.m - GaBP algorithm for dense matrices, parallel version

algorithm described in:
Linear Detection via Belief Propagation
By Danny Bickson, Danny Dolev, Ori Shental, Paul H. Siegel and Jack K. Wolf.
In the 45th Annual Allerton Conference on Communication, Control and Computing, Allerton House, Illinois, 
Sept. 07'

1a) run_gabp.m - example script for running gabp.m
2) asynch_GBP.m - GaBP algorithm for dense matrices, serial (asynch. version)
3) sparse_gabp.m - GaBP algorithm for sparse matrices, optimized.
This code was tested with instances of size 500,000 x 500,000 with 4% non zeros
3a) run_sparse_gabp.m - script file for running sprase_gabp.m
4) gabpms.m - Min-Sum algorithm
C.C. Moallemi and B. Van Roy. “Convergence of the min-sum algorithm for convex optimization.”
4a) run_gabpms.m - script for running the gabpms.m file
5) gabp_inv.m - runs GaBP for inverting a matrix
5a) run_gabp_inv.m - script for running gabp_inv
6) blockGBP.m, test_blockGBP.m - run block version of GaBP (vectoric version),

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The following directories include samples of problems solved using GaBP.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
1) LDLC - extended low density lattice decoder. Using parametric
representation of Gaussian mixtures.
This codes needs the KDE Matlab package by A. T. Ihler, found in http://ttic.uchicago.edu/~ihler/code/
After unpacking, you should addpath() in Matlab to the root kde directory.
The code is given for educational purposes, and does not scale beyond very
small examples.

2) GaBP_convergence_fix - fixing the convergence of the GaBP algorithm
"Fixing the convergence of the GaBP algorithm" by J. K. Johnson, D. Bickson and D. Dolev
In IEEE Internatioanal Synposium on Information Theory, Seoul, South Korea, July 2009. 
http://arxiv.org/abs/0901.4192

3) LP - linear programming example
"Polynomial Linear Programming with Gaussian Belief Propagation. By Danny Bickson, Yoav Tock, 
Ori Shental, Paul H. Seigel, Jack K. Wolf and Danny Dolev. In the Forty-Sixth Annual Allerton 
Conference on Communication, Control, and Computing, Sept. 2008, Allerton House, Monticello, Illinois.
http://arxiv.org/abs/0810.1631

4) NUM - network utility maximizationa - interior point example
"Distributed Large Scale Network Utility Maximization", by D. Bickson, Y. Tock, A. Zymnis, S. Boyd and D. Dolev.
In IEEE Internatioanal Synposium on Information Theory, Seoul, South Korea, July 2009.  
http://arxiv.org/abs/0901.2684

5) NBP - non-parametric belief propagation implementation (using discretization) - more efficient.
This code contains examples of both compressive sensing and low density
lattice decoding. The code was tested with sparse decoding matrices up to size
100,000 x 100,000.
Algorithm description is available: D. Baron, S. Sarvotham, and R. G. Baraniuk, "Bayesian Compressive Sensing via Belief Propagation," IEEE 
Transactions on Signal Processing.

6) Sensor selection example.
Distributed sensor selection via Gaussian belief propagation. D. Bickson and D. Dolev.
Manuscript in preparation.
http://arxiv.org/abs/0907.0931

7) NBP decoder/ - "A low density lattice decoder via non-parametric belief propagation"
By D. Bickson, A. T. Ihler, H. Avissar and D. Dolev.
In the 47th Allerton Communication Control and Computing Conference, Sept.
2009, Allerton House, IL.  http://arxiv.org/abs/0901.3197

8) gabp-conv/ - "Message Passing Multi-user Detection. D. Bickson and C. Guestrin.
Manuscript in preperation.

9) fault_detection/ - code for fault identification.
Fault Identification via Non-parametric Belief Propagation. By Danny Bickson,
Dror Baron, Alex Ihler, Harel Avissar and Danny Dolev. Submitted to IEEE Tran.
On Signal Processing. http://arxiv.org/abs/0908.2005

