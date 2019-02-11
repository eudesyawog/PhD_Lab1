library(ggplot2)
library (readr)

df <- read.csv("/home/je/Bureau/per_class_f1score.csv",colClasses = c("factor","numeric","character"))

df$Classifier[df$Classifier=="EMP99"]<-"RF(PCs,MPs)"
df$Classifier[df$Classifier=="Spectrals Bands"]<-"RF(SF)"
df$Classifier[df$Classifier=="EMP99 + Spectral Bands"]<-"RF(SF,PCs,MPs)"

p<-ggplot(df, aes(x = Class, y = F1.score, fill=Classifier))+
  geom_bar(stat="identity",position=position_dodge()) + theme_bw()
p + scale_y_continuous(name="F1 score")
ggsave("/home/je/Bureau/per_class_f1_score_reunion.png", width=7.18,height=5,dpi = 300)

