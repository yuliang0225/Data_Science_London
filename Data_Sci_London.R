## Data View

# Read Data
traIn <- read.table("Data/train.csv",sep=',',header = F)
traOu <- read.table("Data/trainLabels.csv",sep=',',header = F)
tesIn <- read.table("Data/test.csv",sep=',',header = F)
train_df <- as.data.frame(traIn,traOu)
