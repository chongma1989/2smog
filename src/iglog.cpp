// Copyright (c) 2018 - 2020 Chong Ma
// 
// This file contains the kernel function for the R package glog. 
// The function glog is written for the integrative generalized linear model 
// constraint on specified hierarchical structures by using overlapped group 
// penalty. It is implemented by combining the ISTA and ADMM algorithms, and 
// works for continuous, multimonial and survival data. 


//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins("cpp11")]]
//[[Rcpp::interfaces(r,cpp)]]
#include <RcppArmadillo.h>
#include "penalty.h"

using namespace Rcpp;
using namespace arma;

//' Integrative generalized linear model constraint on hierarchical structure
//' by using overlapped group penalty 
//' 
//' @param y1 a survival object contains the survival time and censoring status 
//'           from data1. See \code{\link[survival]{Surv}}.
//' @param x1 the design matrix of n by p from data1. 
//' @param y2 a survival object contains the survival time and censoring status 
//'           from data2. See \code{\link[survival]{Surv}}.
//' @param x2 the design matrix of n by p from data2. x1 and x2 should have 
//'           the same number of columns. 
//' @param g a vector of group labels for the p predictor variables.
//' @param v a vector of 0 and 1 for the penalization status of the 
//'          p predictor variables. 1 is for penalization and 0 for 
//'          not penalization. 
//' @param hierarchy hierarchy indicator. 0 for L2 penalty, 1 for the
//'                  composite L2 penalty, and 2 for the composite L2 
//'                  and ridge penalty for each group, respectively.  
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
//' # generate two design matrices x1 and x2
//' s=10
//' x1=matrix(0,n,1+2*p)
//' x1[,1]=sample(c(0,1),n,replace = TRUE)
//' x1[,seq(2,1+2*p,2)]=matrix(rnorm(n*p),n,p)
//' x1[,seq(3,1+2*p,2)]=x1[,seq(2,1+2*p,2)]*x1[,1]
//' 
//' x2=matrix(0,n,1+2*p)
//' x2[,1]=x1[,1]
//' x2[,seq(2,1+2*p,2)]=matrix(rnorm(n*p),n,p)
//' x2[,seq(3,1+2*p,2)]=x2[,seq(2,1+2*p,2)]*x2[,1] 
//' 
//' # generate beta1 and beta2 
//' beta1=beta2=c(rnorm(13,0,2),rep(0,ncol(x1)-13))
//' beta2[1:13]=beta2[1:13]+rnorm(13,0,0.1)
//' beta1[c(2,4,7,9)]=beta2[c(2,4,7,9)]=0
//' 
//' # generate two continuous y1 and y2
//' ldata1=x1%*%beta1
//' noise1=rnorm(n)
//' snr1=as.numeric(sqrt(var(ldata1)/(s*var(noise1))))
//' ly1=ldata1+snr1*noise1
//' 
//' ldata2=x2%*%beta2
//' noise2=rnorm(n)
//' snr2=as.numeric(sqrt(var(ldata2)/(s*var(noise1))))
//' ly2=ldata2+snr2*noise2 
//' 
//' g=c(p+1,rep(1:p,rep(2,p)))
//' v=c(0,rep(1,2*p))
//' \dontrun{
//' ilfit1=iglog(y1=as.matrix(ly1),x1=as.matrix(x1),
//'              y2=as.matrix(ly2),x2=as.matrix(x2),
//'              g=g,v=v,hierarchy=1,lambda=c(0.01,0.001),
//'              type="lm")
//' }
//' 
//' ## generate two binomial data 
//' prob1=exp(as.matrix(x1)%*%as.matrix(beta1))/(1+exp(as.matrix(x1)%*%as.matrix(beta1)))
//' cy1=ifelse(prob1<0.5,0,1)
//' 
//' prob2=exp(as.matrix(x2)%*%as.matrix(beta2))/(1+exp(as.matrix(x2)%*%as.matrix(beta2)))
//' cy2=ifelse(prob2<0.5,0,1)
//' 
//' \dontrun{
//' ilfit2=iglog(y1=as.matrix(cy1),x1=as.matrix(x1),
//'              y2=as.matrix(cy2),x2=as.matrix(x2),
//'              g=g,v=v,hierarchy=1,lambda=c(0.025,0.001),
//'              type="binomial")
//' }
//' 
//' ## generate two survival data 
//' sdata1=sim.survdata(N=n,T=100,X=x1,beta=beta1)
//' sy1=sdata1$data[,c("y","failed")]
//' sy1$failed=ifelse(sy1$failed,1,0)
//' 
//' sdata2=sim.survdata(N=n,T=100,X=x2,beta=beta2)
//' sy2=sdata2$data[,c("y","failed")]
//' sy2$failed=ifelse(sy2$failed,1,0)
//' 
//' \dontrun{
//' ilfit3=iglog(y1=as.matrix(sy1),x1=as.matrix(x1),
//'              y2=as.matrix(sy2),x2=as.matrix(x2),
//'              g=g,v=v,hierarchy=1,lambda=c(0.075,0.001),
//'              type="survival")
//' }
//'
//[[Rcpp::export]]
Rcpp::List iglog(const arma::mat & y1, 
                 const arma::mat & x1,
                 const arma::mat & y2,
                 const arma::mat & x2,
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
      // centralize y1, y2, x1, x2
      const arma::vec y1_0 = y1.col(0);
      const arma::vec y2_0 = y2.col(0);
      
      const arma::vec Y1 = scale ? y1_0 - arma::mean(y1_0) : y1_0;
      const arma::mat X1 = scale ? x1.each_row() - arma::mean(x1) : x1;
      const arma::vec Y2 = scale ? y2_0 - arma::mean(y2_0) : y2_0;
      const arma::mat X2 = scale ? x2.each_row() - arma::mean(x2) : x2;
      
      //const double L = 1/arma::max(arma::svd(X)); // maximum penalty value
      const int p = x1.n_cols;
      const int n1 = x1.n_rows;
      const int n2 = x2.n_rows;
      
      // cache the inverse matrix
      const arma::mat I_p = arma::eye<arma::mat>(p,p);
      const arma::mat I_n1 = arma::eye<arma::mat>(n1,n1);
      const arma::mat I_n2 = arma::eye<arma::mat>(n2,n2);
      
      const arma::mat invm1 = 1.0/rho*(I_p - X1.t()*(rho*I_n1 + X1*X1.t()).i()*X1);
      const arma::mat invm2 = 1.0/rho*(I_p - X2.t()*(rho*I_n2 + X2*X2.t()).i()*X2);
      
      //initialize primal, dual and error variables
      arma::vec new_beta1 = arma::zeros(p);
      arma::vec new_beta2 = arma::zeros(p);
      
      arma::vec old_Z1 = arma::zeros(p);  // augmented variable for beta1
      arma::vec new_Z1 = arma::zeros(p);
      
      arma::vec old_Z2 = arma::zeros(p);  // augmented variable for beta2
      arma::vec new_Z2 = arma::zeros(p);
      
      arma::vec U1 = arma::zeros(p);      // dual variable for beta1
      arma::vec U2 = arma::zeros(p);      // dual variable for beta1
      
      std::vector<double> epri;   // primal error
      std::vector<double> edual;  // dual error
      double epri_ctr = 0.0;      // primal error control
      double edual_ctr = 0.0;     // dual error control
      
      const arma::uvec idx0 = arma::find(v == 0);
      arma::uvec ug = arma::unique(g%v);
      arma::uvec upg = ug(arma::find(ug>0)); // groups for penalization
      
      // joint modeling: hierarchical structure with L2 penalty
      // gene_expr + mutation + gene_expr*treatment + mutation*treatment
      std::vector<double> llike;
      int itr = 0;
      do{
        double lpenalty = 0.0;
        // ADMM: update the primal variable -- beta
        new_beta1 = invm1*(X1.t()*Y1+rho*(new_Z1-U1));
        new_beta2 = invm2*(X2.t()*Y2+rho*(new_Z2-U2));
        
        // ADMM: update the dual variable -- Z
        new_Z1.elem(idx0) = new_beta1.elem(idx0) + U1.elem(idx0);
        new_Z2.elem(idx0) = new_beta2.elem(idx0) + U2.elem(idx0);
        
        arma::uvec idx1;
        arma::vec temp_Z;
        arma::uvec idx2 = arma::regspace<arma::uvec>(1,2,3);
        for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
          idx1 = arma::find(g == (*it));
          
          // join the gene_expre, gene_expre*trt and mutation, mutation*trt
          temp_Z = arma::join_cols(new_beta1.elem(idx1)+U1.elem(idx1),
                                   new_beta2.elem(idx1)+U2.elem(idx1));
          temp_Z = prox(temp_Z,lambda/rho,hierarchy,idx2);
          lpenalty += penalty(temp_Z,lambda,hierarchy,idx2);
         
          new_Z1.elem(idx1) = temp_Z.subvec(0,1);
          new_Z2.elem(idx1) = temp_Z.subvec(2,3);
        }
        
        // ADMM: update the dual variable -- U
        U1=U1+(new_beta1 - new_Z1);
        U2=U2+(new_beta2 - new_Z2);
        
        // ADMM: Update primal and dual errors
        arma::vec new_beta = arma::join_cols(new_beta1,new_beta2);
        arma::vec new_Z = arma::join_cols(new_Z1,new_Z2);
        arma::vec old_Z = arma::join_cols(old_Z1,old_Z2);
        arma::vec U = arma::join_cols(U1,U2);
        
        epri.push_back(arma::norm(new_beta - new_Z,2)/std::sqrt(2*p));
        edual.push_back(arma::norm(new_Z - old_Z,2)/std::sqrt(2*p));
        
        epri_ctr = eabs + erel/std::sqrt(2*p)*(arma::norm(new_beta,2)>arma::norm(new_Z,2) ? 
                                                 arma::norm(new_beta,2) : arma::norm(new_Z,2));
        edual_ctr = std::sqrt((n1+n2)/(2*p))*eabs/rho + erel/std::sqrt(2*p)*(arma::norm(U,2));
        
        old_Z1 = new_Z1;
        old_Z2 = new_Z2;
        llike.push_back(0.0 - std::pow(arma::norm(Y1 - X1*new_Z1,2),2)/2 - 
                            - std::pow(arma::norm(Y2 - X2*new_Z2,2),2)/2 - lpenalty);
        itr++;
      } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
      
      arma::uvec subindex1 = arma::find(arma::abs(new_Z1) > 0.0);
      arma::uvec subindex2 = arma::find(arma::abs(new_Z2) > 0.0);
      
      arma::vec beta1 = new_Z1.elem(subindex1);
      arma::vec beta2 = new_Z2.elem(subindex2);
      
      int nsubidx1 = subindex1.n_elem;
      int nsubidx2 = subindex2.n_elem;
      
      arma::rowvec xm1 = arma::mean(x1);
      double beta1_0 = subindex1.is_empty() ? arma::mean(y1_0) : 
                                              arma::mean(y1_0) - arma::dot(xm1.elem(subindex1),beta1);
      
      arma::rowvec xm2 = arma::mean(x2);
      double beta2_0 = subindex2.is_empty() ? arma::mean(y2_0) : 
                                              arma::mean(y2_0) - arma::dot(xm2.elem(subindex2),beta2);
      
      arma::uvec Beta1(nsubidx1+1);
      arma::vec Coefficients1(nsubidx1+1);
      
      arma::uvec Beta2(nsubidx2+1);
      arma::vec Coefficients2(nsubidx2+1);
      
      if(nsubidx1 == 0){
        Beta1(0) = 0;
        Coefficients1(0) = beta1_0;
      }else{
        Beta1(0) = 0;
        Beta1.subvec(1,nsubidx1) = subindex1 + 1;
        
        Coefficients1(0) = beta1_0;
        Coefficients1.subvec(1,nsubidx1) = beta1;
      }
      
      if(nsubidx2 == 0){
        Beta2(0) = 0;
        Coefficients2(0) = beta2_0;
      }else{
        Beta2(0) = 0;
        Beta2.subvec(1,nsubidx2) = subindex2 + 1;
        
        Coefficients2(0) = beta2_0;
        Coefficients2.subvec(1,nsubidx2) = beta2;
      }
      
      return Rcpp::List::create(Rcpp::Named("model") = 
                                Rcpp::DataFrame::create(_["Beta"] = Beta1,
                                                        _["Coefficients"] = Coefficients1,
                                                        _["Coefficients"] = Coefficients2),
                                                        Rcpp::Named("loglike") = llike,
                                                        Rcpp::Named("PrimalError") = epri,
                                                        Rcpp::Named("DualError") = edual,
                                                        Rcpp::Named("converge") = itr);
  
  }else if(type == "binomial"){
    
    const arma::vec y1_0 = y1.col(0);
    const arma::vec y2_0 = y2.col(0);
    
    // centralize x1 and x2
    const arma::mat X1 = scale ? x1.each_row() - arma::mean(x1) : x1;
    const arma::mat X2 = scale ? x2.each_row() - arma::mean(x2) : x2;
    
    const arma::vec G1 = arma::unique(y1_0);  // group levels 
    const int K1 = G1.n_elem;               // number of groups
    
    const arma::vec G2 = arma::unique(y2_0);  // group levels 
    const int K2 = G2.n_elem;               // number of groups
    
    if(K1!=K2 || arma::any(G1 != G2)){
      Rcpp::stop("y1 and y2 must have the same groups");
    }
    
    const int p = X1.n_cols;
    const int n1 = X1.n_rows;
    const int n2 = X2.n_rows;
    
    //initialize primal, dual and error variables
    arma::mat new_beta1 = arma::zeros<arma::mat>(p,K1-1);
    arma::mat new_beta2 = arma::zeros<arma::mat>(p,K2-1);
    
    arma::mat old_Z1 = arma::zeros<arma::mat>(p,K1-1);  // augmented variable for beta1
    arma::mat new_Z1 = arma::zeros<arma::mat>(p,K1-1);
    
    arma::mat old_Z2 = arma::zeros<arma::mat>(p,K2-1);  // augmented variable for beta2
    arma::mat new_Z2 = arma::zeros<arma::mat>(p,K2-1);
    
    arma::mat U1 = arma::zeros<arma::mat>(p,K1-1);      // dual variable for beta1
    arma::mat U2 = arma::zeros<arma::mat>(p,K2-1);      // dual variable for beta1
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    const arma::uvec idx0 = arma::find(v == 0);
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0)); // groups for penalization
    
    // for data1
    arma::mat theta1 = arma::zeros<arma::mat>(n1,K1-1);                 // exp(x*old_beta)
    arma::mat temp_theta1 = arma::zeros<arma::mat>(n1,K1-1);            // exp(x*new_beta)
    arma::mat grad1 =  arma::zeros<arma::mat>(p,K1-1);                 // gradient 
    arma::mat CX1 = arma::zeros<arma::mat>(p,K1-1);
    for(int k = 0; k<K1-1; ++k){
      CX1.col(k) = arma::vectorise(arma::sum(X1.rows(arma::find(y1_0==G1(k)))));
    }
    
    // for data2
    arma::mat theta2 = arma::zeros<arma::mat>(n2,K2-1);                 // exp(x*old_beta)
    arma::mat temp_theta2 = arma::zeros<arma::mat>(n2,K2-1);            // exp(x*new_beta)
    arma::mat grad2 =  arma::zeros<arma::mat>(p,K2-1);                 // gradient 
    arma::mat CX2 = arma::zeros<arma::mat>(p,K2-1);
    for(int k = 0; k<K2-1; ++k){
      CX2.col(k) = arma::vectorise(arma::sum(X2.rows(arma::find(y2_0==G2(k)))));
    }
    
    double L1 = LL;
    double L2 = LL;
    
    std::vector<double> llike;
    
    // joint modeling: hierarchical structure with L2 penalty
    // gene_expr + mutation + gene_expr*treatment + mutation*treatment
    int itr = 0;
    do{
      double old_f1 = 0.0;
      double new_f1 = 0.0;
      double old_f2 = 0.0;
      double new_f2 = 0.0;
      double Q1 = 0.0;
      double Q2 = 0.0;
      double lpenalty = 0.0;
      
      //**********************************************************************************//
      // Majorization-Minization step for beta1 (use FISTA)
      theta1 = arma::exp(X1*new_beta1);        // dictionary for loglike fun and gradient
      grad1=X1.t()*((1.0/(1+arma::sum(theta1,1)))%theta1.each_col());
      old_f1 = arma::sum(arma::vectorise(arma::log(1+arma::sum(theta1,1))));
      
      // inner loop for FASTA
      int inner_itr1 = 0;
      arma::mat temp_beta1 = arma::zeros<arma::mat>(p,K1-1);
      do{
        // ADMM: update the primal variable -- beta
        temp_beta1 = 1.0/(L1+rho)*(L1*new_beta1-grad1+CX1+rho*(new_Z1-U1));
        temp_theta1 = arma::exp(X1*temp_beta1);        // dictionary for loglike fun, gradient and hessian matrix
        new_f1 = arma::sum(arma::vectorise(arma::log(1+arma::sum(temp_theta1,1))));
        Q1 = old_f1 + arma::dot(arma::vectorise(grad1),arma::vectorise(temp_beta1-new_beta1)) +
          L1/2*arma::norm(temp_beta1-new_beta1,2);
        L1 *= eta;
        inner_itr1++;
      } while (new_f1 - Q1 > 0 && inner_itr1<50);
      new_beta1 = temp_beta1;
      
      //**********************************************************************************//
      // Majorization-Minization step for beta2 (use FISTA)
      theta2 = arma::exp(X2*new_beta2);        // dictionary for loglike fun and gradient
      grad2 = X2.t()*((1.0/(1+arma::sum(theta2,1)))%theta2.each_col());
      old_f2 = arma::sum(arma::vectorise(arma::log(1+arma::sum(theta2,1))));
      
      // inner loop for FASTA
      int inner_itr2 = 0;
      arma::mat temp_beta2 = arma::zeros<arma::mat>(p,K2-1);
      do{
        // ADMM: update the primal variable -- beta
        temp_beta2 = 1.0/(L2+rho)*(L2*new_beta2-grad2+CX2+rho*(new_Z2-U2));
        temp_theta2 = arma::exp(X2*temp_beta2);        // dictionary for loglike fun, gradient and hessian matrix
        new_f2 = arma::sum(arma::vectorise(arma::log(1+arma::sum(temp_theta2,1))));
        Q2 = old_f2 + arma::dot(arma::vectorise(grad2),arma::vectorise(temp_beta2-new_beta2)) +
          L2/2*arma::norm(temp_beta2-new_beta2,2);
        L2 *= eta;
        inner_itr2++;
      } while (new_f2 - Q2 > 0 && inner_itr2<50);
      new_beta2 = temp_beta2;
      
      //*************************************************************************************************//
      // ADMM: update the dual variable -- Z
      arma::uvec uk;
      for(int k = 0; k<K1-1; ++k){
        uk << k;
        new_Z1(idx0,uk) = new_beta1(idx0,uk) + U1(idx0,uk);
        new_Z2(idx0,uk) = new_beta2(idx0,uk) + U2(idx0,uk);
        
        arma::uvec idx1;
        arma::vec temp_Z;
        arma::uvec idx2 = arma::regspace<arma::uvec>(1,2,3);
        for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
          idx1 = arma::find(g == (*it));
          
          // join the gene_expre, gene_expre*trt and mutation, mutation*trt
          temp_Z = arma::join_cols(new_beta1(idx1,uk)+U1(idx1,uk),
                                   new_beta2(idx1,uk)+U2(idx1,uk));
          temp_Z = prox(temp_Z,lambda/rho,hierarchy,idx2);
          lpenalty += penalty(temp_Z,lambda,hierarchy,idx2);
       
          new_Z1(idx1,uk) = temp_Z.subvec(0,1);
          new_Z2(idx1,uk) = temp_Z.subvec(2,3);
        }
      }
      
      //*************************************************************************************************//
      // ADMM: update the dual variable -- U
      U1=U1+(new_beta1 - new_Z1);
      U2=U2+(new_beta2 - new_Z2);
      
      //*************************************************************************************************//
      // ADMM: Update primal and dual errors
      epri.push_back(arma::norm(arma::join_cols(arma::vectorise(new_beta1 - new_Z1),
                                                arma::vectorise(new_beta2 - new_Z2)),2)/std::sqrt(p*(K1+K2-2)));
      
      edual.push_back(arma::norm(arma::join_cols(arma::vectorise(new_Z1 - old_Z1),
                                                 arma::vectorise(new_Z2 - old_Z2)),2)/std::sqrt(p*(K1+K2-2)));
      
      epri_ctr = eabs + erel/std::sqrt(p*(K1+K2-2))*(arma::norm(arma::vectorise(arma::join_cols(new_beta1,new_beta2)),2) > 
                                                       arma::norm(arma::vectorise(arma::join_cols(new_Z1,new_Z2)),2) ? 
                                                       arma::norm(arma::vectorise(arma::join_cols(new_beta1,new_beta2)),2) : 
                                                       arma::norm(arma::vectorise(arma::join_cols(new_Z1,new_Z2)),2));
      
      edual_ctr = std::sqrt((n1+n2)/(p*(K1+K2-2)))*eabs/rho + erel/std::sqrt(p*(K1+K2-2))*
        (arma::norm(arma::vectorise(arma::join_cols(U1,U2)),2));
      
      old_Z1 = new_Z1;
      old_Z2 = new_Z2;
      llike.push_back(arma::sum(arma::sum(CX1%new_beta1)) - new_f1 + 
                      arma::sum(arma::sum(CX2%new_beta2)) - new_f2 - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    // get the selected variables and the corresponding coefficients
    // for data1
    arma::uvec subindex1;
    arma::uvec Beta1;
    arma::mat Coefficients1;
    
    // for data2
    arma::uvec subindex2;
    arma::uvec Beta2;
    arma::mat Coefficients2;
    
    arma::uvec uk;
    for(int k = 0; k<K1-1; ++k){
      uk << k;
      subindex1 = arma::join_cols(subindex1,arma::find(arma::abs(new_Z1.col(k)) > 0.0));
      subindex2 = arma::join_cols(subindex2,arma::find(arma::abs(new_Z2.col(k)) > 0.0));
    }
    Beta1 = arma::unique(subindex1);
    Coefficients1 = new_Z1.rows(Beta1);
    
    Beta2 = arma::unique(subindex2);
    Coefficients2 = new_Z2.rows(Beta2);
    
    return Rcpp::List::create(Rcpp::Named("model")=Rcpp::DataFrame::create(_["Beta"] = Beta1+1,
                                          _["Coefficients1"] = Coefficients1,
                                          _["Coefficients2"] = Coefficients2),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
    
  }else if(type == "survival"){
    // centralize x1 and x2
    const arma::mat X1 = scale ? x1.each_row() - arma::mean(x1) : x1; 
    const arma::mat X2 = scale ? x2.each_row() - arma::mean(x2) : x2; 
    
    const arma::vec time1 = y1.col(0);
    const arma::vec censor1 = y1.col(1);
    arma::uvec idx0_1 = arma::find(censor1>0);
    arma::vec Y1 = arma::unique(time1(idx0_1));
    
    const arma::vec time2 = y2.col(0);
    const arma::vec censor2 = y2.col(1);
    arma::uvec idx0_2 = arma::find(censor2>0);
    arma::vec Y2 = arma::unique(time2(idx0_2));
    
    //const double L = 1/arma::max(arma::svd(X)); // maximum penalty value
    const int p = x1.n_cols;
    const int n1 = x1.n_rows;
    const int n2 = x2.n_rows;
    
    //initialize primal, dual and error variables
    arma::vec new_beta1 = arma::zeros(p);
    arma::vec new_beta2 = arma::zeros(p);
    
    arma::vec old_Z1 = arma::zeros(p);  // augmented variable for beta1
    arma::vec new_Z1 = arma::zeros(p);
    
    arma::vec old_Z2 = arma::zeros(p);  // augmented variable for beta2
    arma::vec new_Z2 = arma::zeros(p);
    
    arma::vec U1 = arma::zeros(p);      // dual variable for beta1
    arma::vec U2 = arma::zeros(p);      // dual variable for beta1
    
    std::vector<double> epri;   // primal error
    std::vector<double> edual;  // dual error
    double epri_ctr = 0.0;      // primal error control
    double edual_ctr = 0.0;     // dual error control
    
    const arma::uvec idx0 = arma::find(v == 0);
    arma::uvec ug = arma::unique(g%v);
    arma::uvec upg = ug(arma::find(ug>0)); // groups for penalization
    
    // for data1
    arma::vec theta1 = arma::zeros(n1);
    arma::vec temp_theta1 = arma::zeros(n1);                 // exp(x*new_beta)
    arma::mat grad_dic1 = arma::zeros<arma::mat>(p,n1);
    arma::vec grad1 =  arma::zeros(p);
    
    // for data2
    arma::vec theta2 = arma::zeros(n2);
    arma::vec temp_theta2 = arma::zeros(n2);                 // exp(x*new_beta)
    arma::mat grad_dic2 = arma::zeros<arma::mat>(p,n2);
    arma::vec grad2 =  arma::zeros(p);
    
    double L1 = LL;
    double L2 = LL;
    
    double res1;
    double res2;
    std::vector<double> llike;
    
    // joint modeling: hierarchical structure with L2 penalty
    // gene_expr + mutation + gene_expr*treatment + mutation*treatment
    int itr = 0;
    do{
      // ADMM: update the primal variable -- beta
      arma::uvec temp0;
      arma::uvec temp1;
      double old_f1 = 0.0;
      double old_f2 = 0.0;
      double lpenalty = 0.0;
      
      //*************************************************************************************************//
      // Majorization-Minimization for beta1
      // dictionary for loglike fun, gradient and hessian matrix
      theta1 = arma::exp(X1*new_beta1);        // dictionary for loglike fun, gradient and hessian matrix
      grad_dic1 = (X1.each_col()%theta1).t();
      for(arma::vec::iterator it = Y1.begin(); it!=Y1.end(); ++it){
        temp0 = arma::find(time1 == (*it));  // tied survival time with noncensoring status
        temp1 = arma::find(time1 > (*it));  // risk observations at time *it (include censored observations)
        
        // inner loop for tied observation
        int temp0_n = temp0.n_elem;
        double cnst = 0.0;
        for(int l = 0; l<temp0_n; ++l){
          cnst = arma::sum(theta1.elem(temp0))*(1-l/temp0_n)+arma::sum(theta1.elem(temp1));
          old_f1 += std::log(cnst);
          grad1 += 1.0/cnst*(arma::sum(grad_dic1.cols(temp0),1)*(1-l/temp0_n) + arma::sum(grad_dic1.cols(temp1),1));
        }
        old_f1 -= arma::dot(arma::sum(X1.rows(temp0)),new_beta1);
        grad1 -= arma::vectorise(arma::sum(X1.rows(temp0)));
      }
      
      // inner loop for FASTA
      int inner_itr1 = 0;
      arma::vec temp_beta1 = arma::zeros(p);
      double new_f1 = 0.0;
      do{
        // ADMM: update the primal variable -- beta
        temp_beta1 = 1.0/(L1+rho)*(L1*new_beta1-grad1+rho*(new_Z1-U1));
        
        arma::uvec ttemp0;
        arma::uvec ttemp1;
        double Q1 = 0.0;
        new_f1 = 0.0;
        
        // calculate the majorization value
        temp_theta1 = arma::exp(X1*temp_beta1);        // dictionary for loglike fun, gradient and hessian matrix
        for(arma::vec::iterator it = Y1.begin(); it!=Y1.end(); ++it){
          ttemp0 = arma::find(time1 == (*it));  // tied survival time with noncensoring status
          ttemp1 = arma::find(time1 > (*it));  // risk observations at time *it (include censored observations)
          
          // inner loop for tied observation
          int ttemp0_n = ttemp0.n_elem;
          for(int l = 0; l<ttemp0_n; ++l){
            new_f1 += std::log(arma::sum(temp_theta1.elem(ttemp0))*(1-l/ttemp0_n)+arma::sum(temp_theta1.elem(ttemp1)));
          }
          new_f1 -= arma::dot(arma::sum(X1.rows(ttemp0)),new_beta1);
        }
        Q1 = old_f1 + arma::dot(grad1,temp_beta1-new_beta1)+L1/2*arma::norm(temp_beta1-new_beta1,2);
        
        res1 = new_f1 - Q1;
        L1 *= eta;
        inner_itr1++;
      } while (res1 > 0 && inner_itr1<50);
      new_beta1 = temp_beta1;
      
      
      //*************************************************************************************************//
      // Majorization-Minimization for beta2
      // dictionary for loglike fun, gradient and hessian matrix
      theta2 = arma::exp(X2*new_beta2);        // dictionary for loglike fun, gradient and hessian matrix
      grad_dic2 = (X2.each_col()%theta2).t();
      for(arma::vec::iterator it = Y2.begin(); it!=Y2.end(); ++it){
        temp0 = arma::find(time2 == (*it));  // tied survival time with noncensoring status
        temp1 = arma::find(time2 > (*it));  // risk observations at time *it (include censored observations)
        
        // inner loop for tied observation
        int temp0_n = temp0.n_elem;
        double cnst = 0.0;
        for(int l = 0; l<temp0_n; ++l){
          cnst = arma::sum(theta2.elem(temp0))*(1-l/temp0_n)+arma::sum(theta2.elem(temp1));
          old_f2 += std::log(cnst);
          grad2 += 1.0/cnst*(arma::sum(grad_dic2.cols(temp0),1)*(1-l/temp0_n) + arma::sum(grad_dic2.cols(temp1),1));
        }
        old_f2 -= arma::dot(arma::sum(X2.rows(temp0)),new_beta2);
        grad2 -= arma::vectorise(arma::sum(X2.rows(temp0)));
      }
      
      // inner loop for FASTA
      int inner_itr2 = 0;
      arma::vec temp_beta2 = arma::zeros(p);
      double new_f2 = 0.0;
      do{
        // ADMM: update the primal variable -- beta
        temp_beta2 = 1.0/(L2+rho)*(L2*new_beta2-grad2+rho*(new_Z2-U2));
        
        arma::uvec ttemp0;
        arma::uvec ttemp1;
        double Q2 = 0.0;
        new_f2 = 0.0;
        
        // calculate the majorization value
        temp_theta2 = arma::exp(X2*temp_beta2);        // dictionary for loglike fun, gradient and hessian matrix
        for(arma::vec::iterator it = Y2.begin(); it!=Y2.end(); ++it){
          ttemp0 = arma::find(time2 == (*it));  // tied survival time with noncensoring status
          ttemp1 = arma::find(time2 > (*it));  // risk observations at time *it (include censored observations)
          
          // inner loop for tied observation
          int ttemp0_n = ttemp0.n_elem;
          for(int l = 0; l<ttemp0_n; ++l){
            new_f2 += std::log(arma::sum(temp_theta2.elem(ttemp0))*(1-l/ttemp0_n)+arma::sum(temp_theta2.elem(ttemp1)));
          }
          new_f2 -= arma::dot(arma::sum(X2.rows(ttemp0)),new_beta2);
        }
        Q2 = old_f2 + arma::dot(grad2,temp_beta2-new_beta2)+L2/2*arma::norm(temp_beta2-new_beta2,2);
        
        res2 = new_f2 - Q2;
        L2 *= eta;
        inner_itr2++;
      } while (res2 > 0 && inner_itr2<50);
      new_beta2 = temp_beta2;
      
      //*************************************************************************************************//
      // ADMM: update the dual variable -- Z
      new_Z1.elem(idx0) = new_beta1.elem(idx0) + U1.elem(idx0);
      new_Z2.elem(idx0) = new_beta2.elem(idx0) + U2.elem(idx0);
      
      arma::uvec idx1;
      arma::vec temp_Z;
      arma::uvec idx2 = arma::regspace<arma::uvec>(1,2,3);
      for(arma::uvec::iterator it = upg.begin(); it!=upg.end(); ++it){
        idx1 = arma::find(g == (*it));
        
        // join the gene_expre, gene_expre*trt and mutation, mutation*trt
        temp_Z = arma::join_cols(new_beta1.elem(idx1)+U1.elem(idx1),
                                 new_beta2.elem(idx1)+U2.elem(idx1));
        temp_Z = prox(temp_Z,lambda/rho,hierarchy,idx2);
        lpenalty += penalty(temp_Z,lambda,hierarchy,idx2);
       
        new_Z1.elem(idx1) = temp_Z.subvec(0,1);
        new_Z2.elem(idx1) = temp_Z.subvec(2,3);
      }
      
      //*************************************************************************************************//
      // ADMM: update the dual variable -- U
      U1=U1+(new_beta1 - new_Z1);
      U2=U2+(new_beta2 - new_Z2);
      
      //*************************************************************************************************//
      // ADMM: Update primal and dual errors
      arma::vec new_beta = arma::join_cols(new_beta1,new_beta2);
      arma::vec new_Z = arma::join_cols(new_Z1,new_Z2);
      arma::vec old_Z = arma::join_cols(old_Z1,old_Z2);
      arma::vec U = arma::join_cols(U1,U2);
      
      epri.push_back(arma::norm(new_beta - new_Z,2)/std::sqrt(2*p));
      edual.push_back(arma::norm(new_Z - old_Z,2)/std::sqrt(2*p));
      
      epri_ctr = eabs + erel/std::sqrt(2*p)*(arma::norm(new_beta,2)>arma::norm(new_Z,2) ? 
                                               arma::norm(new_beta,2) : arma::norm(new_Z,2));
      edual_ctr = std::sqrt((n1+n2)/(2*p))*eabs/rho + erel/std::sqrt(2*p)*(arma::norm(U,2));
      
      old_Z1 = new_Z1;
      old_Z2 = new_Z2;
      llike.push_back(0.0 - new_f1 - new_f2 - lpenalty);
      itr++;
    } while((epri[itr-1] > epri_ctr || edual[itr-1] > edual_ctr) && itr < maxitr);
    
    // get the selected variables and the corresponding coefficients
    // for data1
    arma::uvec subindex1 = arma::find(arma::abs(new_Z1) > 0.0);
    arma::uvec Beta1 = subindex1+1;
    arma::vec Coefficients1 = new_Z1.elem(subindex1);
    
    // for data2
    arma::uvec subindex2 = arma::find(arma::abs(new_Z2) > 0.0);
    arma::uvec Beta2 = subindex2+1;
    arma::vec Coefficients2 = new_Z2.elem(subindex2);
    
    return Rcpp::List::create(Rcpp::Named("model")=Rcpp::DataFrame::create(_["Beta"] = Beta1,
                                          _["Coefficients1"] = Coefficients1,
                                          _["Coefficients2"] =  Coefficients2),
                                          Rcpp::Named("loglike") = llike,
                                          Rcpp::Named("PrimalError") = epri,
                                          Rcpp::Named("DualError") = edual,
                                          Rcpp::Named("converge") = itr);
  }else {
    Rcpp::stop("type not matched! type must be either lm, binomial or survival.");
  }
  
}

