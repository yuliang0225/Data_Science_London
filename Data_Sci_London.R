## Data View
library(ggplot2)
# Read Data
traIn <- read.table("Data/train.csv",sep=',',header = F)
traOu <- read.table("Data/trainLabels.csv",sep=',',header = F)
tesIn <- read.table("Data/test.csv",sep=',',header = F)
train_df <- data.frame(traIn,factor(traOu$V1))

# PCA Plot
PCA_tran <- prcomp(traIn, retx = TRUE, center = TRUE, scale. = T)
summary(PCA_tran)

# plot
qplot(x=PCA_tran$x[,1],y=PCA_tran$x[,2], colour=train_df$factor.traOu.V1.)+
  theme(legend.position="none")+stat_ellipse(lwd=1)
##



