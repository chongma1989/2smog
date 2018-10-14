// Copyright (c) 2018 - 2020 Chong Ma
// 
// This file contains the kernel function for the R package glog. 
// The function glog is written for the generalized linear model constraint 
// on specified hierarchical structures by using overlapped group penalty. 
// It is implemented by combining the ISTA and ADMM algorithms, and works 
// for continuous, multimonial and survival data. 


//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins("cpp11")]]
//[[Rcpp::interfaces(r,cpp)]]
#include <RcppArmadillo.h>
#include "penalty.h"

using namespace Rcpp;
using namespace arma;

//' Generalized linear model constraint on hierarchical structure
//' by using overlapped group penalty
//' 
//' @param y a vector of numeric value for response variable in the 
//'          generalized linear regression. A matrix of n by 2 for 
//'          survival objects. See \code{\link[survival]{Surv}}.
//' @param x the design matrix of n by p. 
//' @param g a vector of group labels for the p predictor variables.
//' @param v a vector of 0 and 1 for the penalization status of the 
//'          p predictor variables. 1 is for penalization and 0 for 
//'          not penalization. 
//' @param hierarchy hierarchy indicator. 0 for L2 penalty, 1 for the
//'                  composite L1 and L2 penalty, and 2 for the 
//'                  composite L1, L2 and ridge penalty for each 
//'                  group, respectively.  
//' @param type character variable, for different linear models based
//'             on the response variable. For continuous response variable,
//'             type is set ``lm''; for multinomial or binary response 
//'             variable, type is set ``binomial''; for survival response 
//'             variable, type is set ``survival'', respectively. 
//' @param lambda penalty parameters, should correspond to the hierarchy
//'               status. 
//' @param rho   The penalty parameter in the ADMM algorithm. Default is 1e-3.
//' @param scale Whether or not scale the design matrix. Default is true.
//' @param eabs  The absolute tolerance in the ADMM algorithm. Default is 1e-3.
//' @param erel  The reletive tolerance in the ADMM algorithm. Default is 1e-3. 
//' @param LL    Initial value for the coefficient of the second-order term in 
//'              the Majorization-Minimization step. 
//' @param eta   gradient step in the FISTA algorithm.
//' @param maxitr The maximum iterations in the ADMM algorithm. Default is 500. 
//'
//' @examples 
//' 
//' require(coxed)
//' n=50;p=1000
//' set.seed(2018)
//' # set design matrix
//' s=10
//' x=matrix(0,n,1+2*p)
//' x[,1]=sample(c(0,1),n,replace = TRUE)
//' x[,seq(2,1+2*p,2)]=matrix(rnorm(n*p),n,p)
//' x[,seq(3,1+2*p,2)]=x[,seq(2,1+2*p,2)]*x[,1]
//' 
//' # set beta 
//' beta=c(rnorm(13,0,2),rep(0,ncol(x)-13))
//' beta[c(2,4,7,9)]=0
//' 
//' # set y
//' data1=x%*%beta
//' noise1=rnorm(n)
//' snr1=as.numeric(sqrt(var(data1)/(s*var(noise1))))
//' y1=data1+snr1*noise1
//' g=c(p+1,rep(1:p,rep(2,p)))
//' v=c(0,rep(1,2*p))
//' \dontrun{
//' lfit1=glog(y=as.matrix(y1),x=as.matrix(x),g=g,v=v,
//'           hierarchy=1,lambda=c(0.01,0.001))
//' }
//' 
//' ## binomial data 
//' prob=exp(as.matrix(x)%*%as.matrix(beta))/(1+exp(as.matrix(x)%*%as.matrix(beta)))
//' y2=ifelse(prob<0.5,0,1)
//' \dontrun{
//' lfit2=glog(y=as.matrix(y2),x=as.matrix(x),g=g,v=v,
//'            hierarchy=1,lambda=c(0.025,0.001))
//' }
//' 
//' ## survival data 
//' data3=sim.survdata(N=n,T=100,X=x,beta=beta)
//' y3=data3$data[,c("y","failed")]
//' y3$failed=ifelse(y3$failed,1,0)
//' 
//' \dontrun{
//' lfit3=glog(y=as.matrix(y3),x=as.matrix(x),g=g,v=v,
//'            hierarchy=1,lambda=c(0.075,0.001),
//'            type="survival")
//' }
//'
//[[Rcpp::export]]
Rcpp::List glog(const arma::mat & y, 
                const arma::mat & x, 
                const arma::uvec & g, 
                const arma::uvec & v,
                const int & hierarchy,
                const arma::vec & lambda, 
                const std::string & type = "lm",
                const double & rho = 1e-3,
                const bool & scale = true,
                const double & eabs = 1e-3,
                const double & erel = 1e-3,
                const double & LL = 100,
                const double & eta = 1.25,
                const int & maxitr = 500){
  
  if(type == "lm"){
    // centralize y and x
    const arma::vec Y0 = y.col(0);
    const arma::vec Y = scale ? Y0 - arma::mean(Y0) : Y0;
    const arma::mat X = scale ? x.each_row() - arma::mean(x) : x; 
    
    const int n = x.n_rows;
    const int p = x.n_cols;
    
    // cache the inverse matrix 
    const arma::mat I_n = arma::eye<arma::mat>(n,n);
    const arma::mat I_p = arma::eye<arma::mat>(p,p);
    const arma::mat invm = 1.0/rho*(I_p-X.t()*((rho*I_n+X*X.t()).i())*X);
    
    //initialize primal, dual and error variables
    arma::vec new_beta = arma::zeros(p);
    arma::vec old_Z = arma::zeros(p);  // augmented variable for beta
    arma::vec new_Z = arma::zeros(p); 
    arma::vec U = arma::zeros(p);      // dual variable
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error 
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    const arma::uvec idx0 = arma::find(v == 0);
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0));
    
    std::vector<double> llike;
    arma::uvec d; 
    d << 1;
    int itr = 0;
    do{
      double lpenalty = 0.0;
      // ADMM: update the primal variable -- beta
      new_beta = invm*(X.t()*Y+rho*(new_Z-U));
      
      // ADMM: update the dual variable -- Z
      new_Z.elem(idx0) = new_beta.elem(idx0) + U.elem(idx0);
      
      arma::uvec idx1;
      for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
        idx1 = arma::find(g == (*it));
        new_Z.elem(idx1) = prox(new_beta.elem(idx1) + U.elem(idx1),lambda/rho, hierarchy, d);
        lpenalty += penalty(new_Z.elem(idx1), lambda, hierarchy, d);
      }
      
      // ADMM: update the dual variable -- U
      U=U+(new_beta - new_Z);
      
      // ADMM: Update primal and dual errors
      epri.push_back(arma::norm(new_beta - new_Z,2)/std::sqrt(p));
      edual.push_back(arma::norm(new_Z - old_Z,2)/std::sqrt(p));
      
      epri_ctr = eabs + erel/std::sqrt(p)*(arma::norm(new_beta,2) > arma::norm(new_Z,2) ? 
                                             arma::norm(new_beta,2) : arma::norm(new_Z,2));
      edual_ctr = std::sqrt(n/p)*eabs/rho + erel/std::sqrt(p)*(arma::norm(U,2));
      
      old_Z = new_Z;
      llike.push_back(0.0 - std::pow(arma::norm(Y - X*new_Z,2),2)/2 - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    arma::uvec subindex = arma::find(arma::abs(new_Z) > 0.0);
    arma::vec beta = new_Z.elem(subindex);
    int nsubidx = subindex.n_elem;
    
    arma::rowvec xm = arma::mean(x);
    double beta0 = subindex.is_empty() ? arma::mean(y.col(0)) : 
      arma::mean(y.col(0)) - arma::dot(xm.elem(subindex),beta); 
    
    arma::uvec Beta(nsubidx+1);
    arma::vec Coefficients(nsubidx+1);
    
    if(nsubidx == 0){
      Beta(0) = 0;
      Coefficients(0) = beta0;
    }else{
      Beta(0) = 0;
      Beta.subvec(1,nsubidx) = subindex + 1;
      
      Coefficients(0) = beta0;
      Coefficients.subvec(1,nsubidx) = beta; 
    }
    
    return Rcpp::List::create(Rcpp::Named("model")=Rcpp::DataFrame::create(_["Beta"] = Beta,
                                          _["Coefficients"] = Coefficients),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
  }else if(type == "binomial"){
    // centralize x
    const arma::vec Y = y.col(0);
    const arma::mat X = scale ? x.each_row() - arma::mean(x) : x; 
    const arma::vec G = arma::unique(Y);  // group levels 
    
    const int K = G.n_elem;               // number of groups
    const int n = X.n_rows;               // number of observations
    const int p = X.n_cols;               // number of predictor variables
    
    // initialize primal, dual and error variables
    arma::mat new_beta = arma::zeros<arma::mat>(p,K-1);
    arma::mat old_Z = arma::zeros<arma::mat>(p,K-1);  // augmented variable for beta
    arma::mat new_Z = arma::zeros<arma::mat>(p,K-1); 
    arma::mat U = arma::zeros<arma::mat>(p,K-1);      // dual variable
    
    // calculate the gradient 
    arma::mat theta = arma::zeros<arma::mat>(n,K-1);                 // exp(x*old_beta)
    arma::mat temp_theta = arma::zeros<arma::mat>(n,K-1);            // exp(x*new_beta)
    arma::mat grad =  arma::zeros<arma::mat>(p,K-1);                 // gradient 
    arma::mat CX = arma::zeros<arma::mat>(p,K-1);
    for(int k = 0; k<K-1; ++k){
      CX.col(k) = arma::vectorise(arma::sum(X.rows(arma::find(Y==G(k)))));
    }
    
    const arma::uvec idx0 = arma::find(v == 0);  // variables for not penalization 
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0));       // variables for penalization
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error 
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    double L = LL;
    std::vector<double> llike;
    
    arma::uvec d; 
    d << 1;
    int itr=0;
    do{
      // Majorization-Minization step for beta (use FISTA)
      // calculate the gradient
      double old_f = 0.0;
      double new_f = 0.0;
      double Q = 0.0;
      double lpenalty = 0.0;
      
      theta = arma::exp(X*new_beta);        // dictionary for loglike fun and gradient
      grad=X.t()*((1.0/(1+arma::sum(theta,1)))%theta.each_col());
      old_f = arma::sum(arma::vectorise(arma::log(1+arma::sum(theta,1))));
      
      // inner loop for FASTA
      int inner_itr = 0;
      arma::mat temp_beta = arma::zeros<arma::mat>(p,K-1);
      do{
        // ADMM: update the primal variable -- beta
        temp_beta = 1.0/(L+rho)*(L*new_beta-grad+CX+rho*(new_Z-U));
        temp_theta = arma::exp(X*temp_beta);        // dictionary for loglike fun, gradient and hessian matrix
        new_f = arma::sum(arma::vectorise(arma::log(1+arma::sum(temp_theta,1))));
        Q = old_f + arma::dot(arma::vectorise(grad),arma::vectorise(temp_beta-new_beta)) +
          L/2*arma::norm(temp_beta-new_beta,2);
        L *= eta;
        inner_itr++;
      } while (new_f - Q > 0 && inner_itr<50);
      new_beta = temp_beta;
      
      // ADMM: update the dual variable -- Z
      arma::uvec uk;
      for(int k = 0; k<K-1; ++k){
        uk << k;
        new_Z(idx0,uk) = new_beta(idx0,uk) + U(idx0,uk);
        
        arma::uvec idx1;
        for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
          idx1 = arma::find(g == (*it));
          new_Z(idx1,uk) = prox(new_beta(idx1,uk)+U(idx1,uk),lambda/rho,hierarchy,d);
          lpenalty += penalty(new_Z(idx1,uk),lambda,hierarchy,d);
        }
      }
      
      // ADMM: update the dual variable -- U
      U=U+(new_beta - new_Z);
      
      // ADMM: Update primal and dual errors
      epri.push_back(arma::norm(arma::vectorise(new_beta - new_Z),2)/std::sqrt(p*(K-1)));
      edual.push_back(arma::norm(arma::vectorise(new_Z - old_Z),2)/std::sqrt(p*(K-1)));
      
      epri_ctr = eabs + erel/std::sqrt(p*(K-1))*(arma::norm(arma::vectorise(new_beta),2) > arma::norm(arma::vectorise(new_Z),2) ? 
                                                   arma::norm(arma::vectorise(new_beta),2) : arma::norm(arma::vectorise(new_Z),2));
      edual_ctr = std::sqrt(n/p*(K-1))*eabs/rho + erel/std::sqrt(p*(K-1))*(arma::norm(arma::vectorise(U),2));
      
      old_Z = new_Z;
      llike.push_back(arma::sum(arma::sum(CX%new_beta)) - new_f - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    // get the selected variables and the corresponding coefficients
    arma::uvec subindex;
    arma::uvec Beta;
    arma::mat Coefficients;
    
    arma::uvec uk;
    for(int k = 0; k<K-1; ++k){
      uk << k;
      subindex = arma::join_cols(subindex,arma::find(arma::abs(new_Z.col(k)) > 0.0));
    }
    Beta = arma::unique(subindex);
    Coefficients = new_Z.rows(Beta);
    
    return Rcpp::List::create(Rcpp::Named("model")=Rcpp::DataFrame::create(_["Beta"] = Beta+1,
                                          _["Coefficients"] = Coefficients),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
  }else if(type == "survival"){
    // centralize x
    const arma::mat X = scale ? x.each_row() - arma::mean(x) : x; 
    
    const arma::vec time = y.col(0);
    const arma::vec censor = y.col(1);
    arma::uvec idx = arma::find(censor>0);
    arma::vec Y = arma::unique(time(idx));
    
    const int n = X.n_rows;
    const int p = X.n_cols;
    
    //initialize primal, dual and error variables
    arma::vec new_beta = arma::zeros(p);
    arma::vec old_Z = arma::zeros(p);  // augmented variable for beta
    arma::vec new_Z = arma::zeros(p); 
    arma::vec U = arma::zeros(p);      // dual variable
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error 
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    const arma::uvec idx0 = arma::find(v == 0);  // variables for not penalization 
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0));       // variables for penalization
    
    arma::vec theta = arma::zeros(n);                 // exp(x*old_beta)
    arma::vec temp_theta = arma::zeros(n);                // exp(x*new_beta)
    arma::mat grad_dic = arma::zeros<arma::mat>(p,n);    // gradient dictionary for each observation
    arma::vec grad =  arma::zeros(p);
    double L = LL;
    
    double res;
    std::vector<double> llike;
    
    arma::uvec d; 
    d << 1;
    int itr=0;
    do{
      // Majorization-Minization step for beta (use FISTA)
      // calculate gradient and hessian matrix
      arma::uvec temp0;
      arma::uvec temp1;
      double old_f = 0.0;
      double lpenalty = 0.0;
      
      // outer loop for noncensoring observations
      theta = arma::exp(X*new_beta);        // dictionary for loglike fun, gradient and hessian matrix
      grad_dic = (X.each_col()%theta).t();
      for(arma::vec::iterator it = Y.begin(); it!=Y.end(); ++it){
        temp0 = arma::find(time == (*it));  // tied survival time with noncensoring status
        temp1 = arma::find(time > (*it));  // risk observations at time *it (include censored observations)
        
        // inner loop for tied observation
        int temp0_n = temp0.n_elem;
        double cnst = 0.0;
        for(int l = 0; l<temp0_n; ++l){
          cnst = arma::sum(theta.elem(temp0))*(1-l/temp0_n)+arma::sum(theta.elem(temp1));
          old_f += std::log(cnst);
          grad += 1.0/cnst*(arma::sum(grad_dic.cols(temp0),1)*(1-l/temp0_n) + 
            arma::sum(grad_dic.cols(temp1),1));
        }
        old_f -= arma::dot(arma::sum(X.rows(temp0)),new_beta);
        grad -= arma::vectorise(arma::sum(X.rows(temp0)));
      }
      
      // inner loop for FASTA
      int inner_itr = 0;
      arma::vec temp_beta = arma::zeros(p);
      double new_f = 0.0;
      do{
        // ADMM: update the primal variable -- beta
        temp_beta = 1.0/(L+rho)*(L*new_beta-grad+rho*(new_Z-U));
        
        arma::uvec ttemp0;
        arma::uvec ttemp1;
        double Q = 0.0;
        new_f = 0.0;
        
        // calculate the majorization value
        temp_theta = arma::exp(X*temp_beta);        // dictionary for loglike fun, gradient and hessian matrix
        for(arma::vec::iterator it = Y.begin(); it!=Y.end(); ++it){
          ttemp0 = arma::find(time == (*it));  // tied survival time with noncensoring status
          ttemp1 = arma::find(time > (*it));  // risk observations at time *it (include censored observations)
          
          // inner loop for tied observation
          int ttemp0_n=ttemp0.n_elem;
          for(int l = 0; l<ttemp0_n; ++l){
            new_f += std::log(arma::sum(temp_theta.elem(ttemp0))*(1-l/ttemp0_n)+
              arma::sum(temp_theta.elem(ttemp1)));
          }
          new_f -= arma::dot(arma::sum(X.rows(ttemp0)),new_beta);
        }
        Q = old_f + arma::dot(grad,temp_beta-new_beta)+L/2*arma::norm(temp_beta-new_beta,2);
        
        res = new_f - Q;
        L *= eta;
        inner_itr++;
      } while (res>0 && inner_itr<50);
      new_beta = temp_beta;
      
      // ADMM: update the dual variable -- Z
      new_Z.elem(idx0) = new_beta.elem(idx0) + U.elem(idx0);
      
      arma::uvec idx1;
      for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
        idx1 = arma::find(g == (*it));
        new_Z.elem(idx1) = prox(new_beta.elem(idx1)+U.elem(idx1),lambda/rho,hierarchy,d);
        lpenalty += penalty(new_Z.elem(idx1),lambda,hierarchy,d);
      }
      
      // ADMM: update the dual variable -- U
      U=U+(new_beta - new_Z);
      
      // ADMM: Update primal and dual errors
      epri.push_back(arma::norm(new_beta - new_Z,2)/std::sqrt(p));
      edual.push_back(arma::norm(new_Z - old_Z,2)/std::sqrt(p));
      
      epri_ctr = eabs + erel/std::sqrt(p)*(arma::norm(new_beta,2)>arma::norm(new_Z,2) ? 
                                             arma::norm(new_beta,2) : arma::norm(new_Z,2));
      edual_ctr = std::sqrt(n/p)*eabs/rho + erel/std::sqrt(p)*(arma::norm(U,2));
      
      old_Z = new_Z;
      llike.push_back(0.0 - new_f - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    // get the selected variables and the corresponding coefficients
    arma::uvec subindex = arma::find(arma::abs(new_Z) > 0.0);
    arma::uvec Beta = subindex+1;
    arma::vec Coefficients = new_Z.elem(subindex);
    
    return Rcpp::List::create(Rcpp::Named("model")=Rcpp::DataFrame::create(_["Beta"] = Beta,
                                          _["Coefficients"] = Coefficients),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
  }else {
    Rcpp::stop("type not matched! type must be either lm, binomial or survival.");
  }
  
}
