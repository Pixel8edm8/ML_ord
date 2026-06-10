# Lasso regularized Cox regression using leave-one-out (LOO) 
# cross validation (CV).
# During each run of LOO CV, 10-fold CV is performed on training set to 
# select the best model. Then the selected model is applied to the single
# held-out test sample to predict death risk.
setwd("C:/2019821topc/2019917_rsf_brca/R_SRC")

rm(list=ls())
library(rbsurv)
library(glmnet)
set.seed(1)
ptm <- proc.time()
#########################################################################################################
#整理保存数据，以后不再麻烦
#MEs_moduleEigengenes= read.csv("MEs_moduleEigengenes.csv",header = T, fill = T, stringsAsFactors=FALSE)
##s2_MEs = ncol(MEs_moduleEigengenes)
#Eigengenes=MEs_moduleEigengenes[,3:s2_MEs]
#erged_data= read.csv("BRCAMerged_clinic_methylation.csv",header = T, fill = T, stringsAsFactors=FALSE)


#erged_data33=erged_data[,33]
#erged_data33[which(erged_data33=="LIVING")] <-0
#erged_data33[which(erged_data33=="DECEASED")] <-1
#erged_data33<-as.numeric(erged_data33)


#erged_data3233=cbind(erged_data[,32],erged_data33)

#mydata3233 = data.matrix(erged_data3233)
#s1_erged_data = nrow(mydata3233)
#s2_erged_data = ncol(mydata3233)
#brca_data<-matrix(nrow=nrow,ncol=ncol)
#brca_data1=0
#brca_data1=cbind(MEs_moduleEigengenes,mydata3233)

#删除时间中的NA 及负值的行Delete NA and negative rows in time
#library(dplyr)
#brca_data2=filter(brca_data1,brca_data1[ ,15]!="NA")
#brca_data=filter(brca_data2,brca_data2[ ,15]>0)
########################
#write.csv(brca_data,file = 'brca_Surv_data.csv')
#######################################################
#mydata1= read.csv("brca_Surv_data.csv",header = F, fill = T, stringsAsFactors=FALSE)
mydata0= read.csv("ypred_rsf_leave_one_out1115_3.csv",header = F, fill = F, stringsAsFactors=FALSE)
mydata1= read.csv("ypred_train_rsf_leave_one_out1115_3.csv",header = F, fill = F, stringsAsFactors=FALSE)
# month1=read.csv("brca_ytime_test830_14.csv",header = F, fill = F, stringsAsFactors=FALSE)
# status1=read.csv("brca_ystatus_test830_14.csv",header = F, fill = F, stringsAsFactors=FALSE)
# mydata_train_median= read.csv("brca_pred778_779_829_12.csv",header = F, fill = F, stringsAsFactors=FALSE)
# 
# mydata00= mydata0[-1,]
mydata0= mydata0[2:477,]
mydata1= mydata1[2:477,]
mydata = data.matrix(mydata0[,2:4])
ypred_train = data.matrix(mydata1[,2:779])

# mydata11= mydata1[-1,]
# ypred_train = data.matrix(mydata11[,1])
# mydata2=mydata1[,2]
# month1=mydata1[,3]
# status1= mydata1[,4] 
# # mydata_train_median=data.matrix(mydata_train_median)
# mydata = data.matrix(mydata2)
# month=data.matrix(month1)
# status=data.matrix(status1)
s1 = nrow(mydata)
s2 = ncol(mydata)
################################################################
mySurv = Surv(mydata[, 2], mydata[,3]);
x = mydata[, 1]
print(s1)
# leave-one-out CV
group = cbind(numeric(s1))
#mv = median(x)
#mv =mydata[, 1]
ind = 1:s1
for(i in 1:s1){
  # cvfit = cv.glmnet(x[ind!=i,], mySurv[ind!=i,], family = "cox", maxit=5000)
  # preTrain = predict(cvfit, newx = x[ind!=i,], s = 0, type="response")
  # print(coef(cvfit,s=0))
  # #print(preTrain)
  mv = median(ypred_train[i])
  preTest = x[i]
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

png(filename = "C:/2019821topc/2019917_rsf_brca/R_SRC/RSF_Survival_leave_one_out1115_23333.png", width = 5.5, height = 5.5,
	units = "cm", res = 300, pointsize = 7)
# plot(fit, mark.time=TRUE, xlab = "leave_one_out_Months", ylab = "Survival", lty = 1:2,
# 	col = 1:2, cex = 0.5) 
plot(fit, mark.time=TRUE, xlab = "Months", ylab = "Survival", main=paste("RSF",sep=""), lty = 1:2,
  col = 1:2, cex = 0.5)
grid()
legend(x = "bottomright", legend = c(leg1, leg2), lty = 1:2,
	col = 1:2, cex = 0.65)
text(10,0.1,paste("p=", formatC(p, format="g", digits = 3), sep = ""),
	pos = 4, cex = 1)
dev.off()

print(proc.time() - ptm)

