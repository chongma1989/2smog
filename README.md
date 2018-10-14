# R package: smog
Structural Modeling by using Overlapped Group Penalty

## Introduction
This R package is built on regularized regression models constriant on specified structures by using overlapped group penalties. It is widely applicable for small n, large p cases, which might deal with continuous response variable, binary or multinomial response variable, and the survival objects including the survival time and censoring status, etc. The kernel functions in the package are coded in C++, and applies and combines the modern algorithms such as ISTA and ADMM algorithms to solve the constrained optimization problems.   


## Install wap
* Download the zip file `smog_1.0.tar.gz`.
* In R console, run `install.packages("smog_1.0.tar.gz",repos=NULL,type="source")`. 

