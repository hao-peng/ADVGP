This is the code for the paper:
Asynchronous Distributed Variational Gaussian Processes for Regression, 
H. Peng, S. Zhe, X. Zhang and Y. Qi. 
Accepted by Thirty-fourth International Conference on Machine Learning (ICML 2017) , August 2017.

**Warning**: this is only an experimental code for illustration purpose

# Intallation guide:
* Copy folder gp/ and file Makefile to ps-lite.
* In ADVGP folder: export CPLUS_INCLUDE_PATH=$PWD/eigen
* Go to folder ps-lite and compile using make/cmake. PS-lite may use wget to download some other softwares. See more information about ps-lite about how to install ps-lite
* Go to folder ps-lite/gp/, test with "run.sh"

# Contact:
For direct contact, please contact Hao (pengh [at] alumni {dot} purdue.edu

# License:
Apache for ps-lite
MPL2 for Eigen
GPL3.0 for ADVGP
