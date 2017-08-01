#Linear regression with noise
rm(list=ls())

library(fGarch)
library(Matrix)
library(matrixcalc)
library(MASS)
library(stabledist)
library(MixedTS)

#Read data
dataset <- read.csv("data_1_1.csv", header = TRUE, sep = ",")
xd <- dataset$x
yd <- dataset$y

#Directly apply linear regression without consider noise(or Gaussian noise)
lmlse <- lm(yd ~ xd)

#Calculate SSR of residual
SSR_lse = sum((lmlse$resid)^2)


#----------------------------------t distribution-----------------------------
#Assume student t distribution for the non-Gaussian noise

#Fitting t distribution to the residuals
fitdistr(lmlse$resid, "t", df = 2)

#Calculate the log-likelihood
logllh_t <- function(par){
-sum(log(dstd(yd - par[1] - par[2]*xd, sd = par[3], nu = par[4])))
}

#Initialize starting point
start_t = c(lmlse$coef[2], lmlse$coef[1], sd(lmlse$resid), 3)

#Maximize the log-likelihood (minimize -logllh)
mle_t = optim(start_t, logllh_t, hessian = TRUE, method = "L-BFGS-B")

#Check convergence
isConv_t <- mle_t$convergence == 0
#Check Hessian matrix to avoid saddle point
isSPD_t <- isSymmetric(mle_t$hessian) & is.positive.definite(mle_t$hessian)

ypred_t <- mle_t$par[1] + mle_t$par[2]*xd
resid_t <- ypred_t - yd
SSR_t = sum(resid_t^2)


#----------------------------------stable distribution--------------------------
#Assume stable distribution for the non-Gaussian noise

logllh_s <- function(par){
-sum(log(dstable(yd - par[1] - par[2]*xd, alpha = par[3], beta = 0, gamma = par[4], delta = 0)))
}

start_s = c(lmlse$coef[2], lmlse$coef[1], 1.5, 1)
mle_s = optim(start_s, logllh_s, hessian = TRUE, method = "L-BFGS-B")

isConv_s <- mle_s$convergence == 0
isSPD_s <- isSymmetric(mle_s$hessian) & is.positive.definite(mle_s$hessian)

ypred_s <- mle_s$par[1] + mle_s$par[2]*xd
resid_s <- ypred_s - yd
SSR_s = sum(resid_s^2)


#----------------------------------------plot--------------------------------------
#Plot fitting curve
plot(main = "Fitting Result", xd, yd, xlab = "x", ylab = "y")
abline(lmlse, col = "red", lwd = 2)
abline(mle_t$par[1:2], col = "blue", lwd = 2)
abline(mle_s$par[1:2], col = "green", lty = 3, lwd = 2)
legend("bottomleft", c("Least square estimator","t distributed noise", "Stable distributed noise"), 
lty=c(1,1,3), lwd = c(2,2,2), col = c("red","blue","green"))


