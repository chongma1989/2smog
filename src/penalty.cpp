//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins("cpp11")]]

#include <RcppArmadillo.h>
#include "penalty.h"

using namespace Rcpp;
using namespace arma;

//' proximal operator on L1 penalty
//' @param x numeric value.
//' @param lambda numeric value for the L1 penalty parameter.
//'  
//[[Rcpp::export]]
double proxL1(const double &x, const double &lambda){
  double res = std::fabs(x) > lambda ? ( x > lambda ? x-lambda : x+lambda ) : 0;
  return res;
}

//' proximal operator on L2 penalty
//' @param x A numeric vector.
//' @param lambda numeric value for the L2 penalty parameter.
//'  
//[[Rcpp::export]]
arma::vec proxL2(const arma::vec &x, const double &lambda){
  double thr = 1 - lambda*std::sqrt(x.n_elem)/arma::norm(x,2);
  if(thr > 0){
    return thr*x;
  }else{
    return arma::zeros(x.n_elem);
  }
}

//' proximal operator on the composite L1, L2, and ridge penalty
//' @param x A numeric vector of two.
//' @param lambda a vector of three penalty parameters. \eqn{\lambda[1]}
//'        is the L2 penalty for x, \eqn{\lambda[2]} is the ridge penalty
//'        for x, and \eqn{\lambda[3]} is the ridge penalty for x[2], respectively.
//' @param hierarchy Indicator variable for 0, 1, 2. 0 is for no overlap, 1 for 
//'        composite L1 and L2 penalty, and 2 for composite L1, L2 and ridge 
//'        penalty, respectively.
//' @param d indices for overlapped variables in x.   
//' 
//[[Rcpp::export]]
arma::vec prox(const arma::vec &x, const arma::vec &lambda, 
               const int &hierarchy, const arma::uvec &d){
  arma::vec res = x;
  switch(hierarchy){
  case 0:{
    res = proxL2(res,lambda[0]);
    break; 
  }
  case 1:{
    res.elem(d) = proxL2(res.elem(d),lambda[1]);
    // res[1] = proxL1(res[1],lambda[1]);
    res = proxL2(res,lambda[0]);
    break;
  }
  case 2:{
    res.elem(d) = proxL2(res.elem(d),lambda[2]);
    // res[1] = proxL1(res[1],lambda[2]);
    double thr = 1 - lambda[0]*std::sqrt(res.n_elem)/arma::norm(res,2);
    if(thr > 0){
      res = 1.0/(1+2*lambda[1])*thr*res;
    }else{
      res = arma::zeros(res.n_elem);
    }
    break;
  }
  default:{
    Rcpp::stop("hierarchy must be a value from 0, 1, 2");
  }
  }
  
  return res;
}


//' Penalty function on the composite L1, L2, and ridge penalty
//' @param x A numeric vector of two.
//' @param lambda a vector of three penalty parameters. \eqn{\lambda[1]}
//'        is the L2 penalty for x, \eqn{\lambda[2]} is the ridge penalty
//'        for x, and \eqn{\lambda[3]} is the L1 penalty for x[2], respectively.
//' @param hierarchy Indicator variable for 0, 1, 2. 0 is for no overlap, 1 for 
//'        composite L1 and L2 penalty, and 2 for composite L1, L2 and ridge 
//'        penalty, respectively.  
//' @param d indices for overlapped variables in x. 
//' 
//[[Rcpp::export]]
double penalty(const arma::vec &x, const arma::vec &lambda, 
               const int &hierarchy, const arma::uvec &d){
  double res;
  switch(hierarchy){
  case 0:{
    res = arma::norm(x,2)*lambda[0];
    break;
  }
  case 1:{
    res = arma::norm(x,2)*lambda[0] + arma::norm(x.elem(d),2)*lambda[1];
    break;
  }
  case 2:{
    res = arma::norm(x,2)*lambda[0] + std::pow(arma::norm(x,2),2)*lambda[1] + arma::norm(x.elem(d),2)*lambda[2];
    break;
  }
  default:{
    Rcpp::stop("hierarchy must be a value from 0, 1, 2");
  }
  }
  
  return res;
}

