library(ggplot2)
library (readr)

df <- read.csv("/home/je/Bureau/per_class_f1score.csv",colClasses = c("character","numeric","character"))


df$Classifier[df$Classifier=="EMP99"]<-"RF(PCs,MPs)"
df$Classifier[df$Classifier=="Spectrals Bands"]<-"RF(SF)"
df$Classifier[df$Classifier=="EMP99 + Spectral Bands"]<-"RF(SF,PCs,MPs)"

df$Classifier <- factor(df$Classifier, levels = c("RF(SF)","RF(PCs,MPs)","RF(SF,PCs,MPs)"))
df$Class <- factor(df$Class, levels = c("1","2","3","4","5","6","7","8","9","10","11"))

p<-ggplot(df, aes(x = Class, y = F1.score, fill=Classifier))+
  geom_bar(stat="identity",position=position_dodge()) + theme_bw()
p + scale_y_continuous(name="F1 score")
ggsave("/home/je/Bureau/per_class_f1_score_reunion.png", width=7.18,height=5,dpi = 300)


df <- read.csv("/home/je/Bureau/per_class_f1score_dordogne.csv",colClasses = c("character","numeric","character"))


df$Classifier[df$Classifier=="EMP99"]<-"RF(PCs,MPs)"
df$Classifier[df$Classifier=="Spectrals Bands"]<-"RF(SF)"
df$Classifier[df$Classifier=="EMP99 + Spectral Bands"]<-"RF(SF,PCs,MPs)"

df$Classifier <- factor(df$Classifier, levels = c("RF(SF)","RF(PCs,MPs)","RF(SF,PCs,MPs)"))

p<-ggplot(df, aes(x = Class, y = F1.score, fill=Classifier))+
  geom_bar(stat="identity",position=position_dodge()) + theme_bw()
p + scale_y_continuous(name="F1 score")
ggsave("/home/je/Bureau/per_class_f1_score_dordogne.png", width=7.18,height=5,dpi = 300)

