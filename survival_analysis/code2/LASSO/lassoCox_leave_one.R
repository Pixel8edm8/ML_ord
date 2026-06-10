# Lasso regularized Cox regression using 10-fold cross validation (CV).
# This program is only for examining which variables are selected by the 
# lasso Cox model. So the 10-fold CV is performed using the whole data set,
# and then the sparse regression coefficients are obtained from the selected 
# model.
setwd("C:/IJCAI2020/code/LASSO")
library("rbsurv")
library("glmnet")

ptm <- proc.time()
set.seed(1)


#mydata = read.table("periments/lassoCox/rdata_imGene_related_norm.txt", header = FALSE)
mydata= read.csv("brca_Surv_data_MM_0_age.csv",header = F, fill = F, stringsAsFactors=FALSE)
mydata = data.matrix(mydata)
mydata =mydata[2:477,]
s1 = nrow(mydata)
s2 = ncol(mydata)
mySurv = Surv(mydata[, 41], mydata[, 42]);
x =cbind( mydata[, 2:13],mydata[, 15:40])

print(s1)
# leave-one-out CV
group = cbind(numeric(s1))
ind = 1:s1
for(i in 1:s1){
  cvfit = cv.glmnet(x[ind!=i,], mySurv[ind!=i,], family = "cox", maxit=5000)
  preTrain = predict(cvfit, newx = x[ind!=i,], s = 0, type="response")
  print(coef(cvfit,s=0))
  #print(preTrain)
  mv = median(preTrain)
  preTest = predict(cvfit, newx = x[ind==i,], s = "lambda.min", type="response")
  if(preTest < mv){
    group[i] = 1
  }else{
    group[i] = 2
  }
  print(i)
}



# logrank
log1 = survdiff(mySurv ~ group)
p = pchisq(log1$chisq, 1, lower.tail=FALSE)
print(p)

# plot KM curve
fit = survfit(mySurv ~ group)
n1 = sum(group==1)
leg1 = paste("Low risk(", n1, ")", sep = "")
n2 = sum(group==2)
leg2 = paste("High risk(", n2, ")", sep = "")

png(filename = "C:/IJCAI2020/code/LASSO/lasso.png", width = 5.5, height = 5.5,
	units = "cm", res = 300, pointsize = 7)
plot(fit, mark.time=TRUE, xlab = "Months", ylab = "Survival", main=paste(" LASSO",sep=""), lty = 1:2,
     col = 1:2, cex = 0.5)
grid()
legend(x = "bottomright", legend = c(leg1, leg2), lty = 1:2,
       col = 1:2, cex = 0.65)
text(10,0.1,paste("p=", formatC(p, format="g", digits = 3), sep = ""),
     pos = 4, cex = 1)
dev.off()

print(proc.time() - ptm)
