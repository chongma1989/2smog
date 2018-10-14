// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#ifndef RCPP_smog_RCPPEXPORTS_H_GEN_
#define RCPP_smog_RCPPEXPORTS_H_GEN_

#include <RcppArmadillo.h>
#include <Rcpp.h>

namespace smog {

    using namespace Rcpp;

    namespace {
        void validateSignature(const char* sig) {
            Rcpp::Function require = Rcpp::Environment::base_env()["require"];
            require("smog", Rcpp::Named("quietly") = true);
            typedef int(*Ptr_validate)(const char*);
            static Ptr_validate p_validate = (Ptr_validate)
                R_GetCCallable("smog", "_smog_RcppExport_validate");
            if (!p_validate(sig)) {
                throw Rcpp::function_not_exported(
                    "C++ function with signature '" + std::string(sig) + "' not found in smog");
            }
        }
    }

    inline Rcpp::List glog(const arma::mat& y, const arma::mat& x, const arma::uvec& g, const arma::uvec& v, const int& hierarchy, const arma::vec& lambda, const std::string& type = "lm", const double& rho = 1e-3, const bool& scale = true, const double& eabs = 1e-3, const double& erel = 1e-3, const double& LL = 100, const double& eta = 1.25, const int& maxitr = 500) {
        typedef SEXP(*Ptr_glog)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_glog p_glog = NULL;
        if (p_glog == NULL) {
            validateSignature("Rcpp::List(*glog)(const arma::mat&,const arma::mat&,const arma::uvec&,const arma::uvec&,const int&,const arma::vec&,const std::string&,const double&,const bool&,const double&,const double&,const double&,const double&,const int&)");
            p_glog = (Ptr_glog)R_GetCCallable("smog", "_smog_glog");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_glog(Shield<SEXP>(Rcpp::wrap(y)), Shield<SEXP>(Rcpp::wrap(x)), Shield<SEXP>(Rcpp::wrap(g)), Shield<SEXP>(Rcpp::wrap(v)), Shield<SEXP>(Rcpp::wrap(hierarchy)), Shield<SEXP>(Rcpp::wrap(lambda)), Shield<SEXP>(Rcpp::wrap(type)), Shield<SEXP>(Rcpp::wrap(rho)), Shield<SEXP>(Rcpp::wrap(scale)), Shield<SEXP>(Rcpp::wrap(eabs)), Shield<SEXP>(Rcpp::wrap(erel)), Shield<SEXP>(Rcpp::wrap(LL)), Shield<SEXP>(Rcpp::wrap(eta)), Shield<SEXP>(Rcpp::wrap(maxitr)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<Rcpp::List >(rcpp_result_gen);
    }

    inline Rcpp::List iglog(const arma::mat& y1, const arma::mat& x1, const arma::mat& y2, const arma::mat& x2, const arma::uvec& g, const arma::uvec& v, const int& hierarchy, const arma::vec& lambda, const std::string& type = "lm", const double& rho = 1e-3, const bool& scale = true, const double& eabs = 1e-3, const double& erel = 1e-3, const double& LL = 100, const double& eta = 1.25, const int& maxitr = 500) {
        typedef SEXP(*Ptr_iglog)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_iglog p_iglog = NULL;
        if (p_iglog == NULL) {
            validateSignature("Rcpp::List(*iglog)(const arma::mat&,const arma::mat&,const arma::mat&,const arma::mat&,const arma::uvec&,const arma::uvec&,const int&,const arma::vec&,const std::string&,const double&,const bool&,const double&,const double&,const double&,const double&,const int&)");
            p_iglog = (Ptr_iglog)R_GetCCallable("smog", "_smog_iglog");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_iglog(Shield<SEXP>(Rcpp::wrap(y1)), Shield<SEXP>(Rcpp::wrap(x1)), Shield<SEXP>(Rcpp::wrap(y2)), Shield<SEXP>(Rcpp::wrap(x2)), Shield<SEXP>(Rcpp::wrap(g)), Shield<SEXP>(Rcpp::wrap(v)), Shield<SEXP>(Rcpp::wrap(hierarchy)), Shield<SEXP>(Rcpp::wrap(lambda)), Shield<SEXP>(Rcpp::wrap(type)), Shield<SEXP>(Rcpp::wrap(rho)), Shield<SEXP>(Rcpp::wrap(scale)), Shield<SEXP>(Rcpp::wrap(eabs)), Shield<SEXP>(Rcpp::wrap(erel)), Shield<SEXP>(Rcpp::wrap(LL)), Shield<SEXP>(Rcpp::wrap(eta)), Shield<SEXP>(Rcpp::wrap(maxitr)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<Rcpp::List >(rcpp_result_gen);
    }

}

#endif // RCPP_smog_RCPPEXPORTS_H_GEN_