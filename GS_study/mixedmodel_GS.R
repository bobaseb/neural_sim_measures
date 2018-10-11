#!/usr/bin/r

#system("clear")
rm(list=ls())
#options(warn=1)  # print warnings as they occur
#options(warn=2)  # treat warnings as errors

library(lme4)

mack = 1 #1 uses mack, 0 uses op de beeck
clf="svm"

if (mack==1) {
dataset="mack"
} else {
  dataset="odb"
}

d <- read.csv(paste("/home/sebastian/Documents/odb_results/results/",dataset,"_slimified.csv",sep=""))

roi_labels <- c(list.files(path = '/home/sebastian/Documents/fiftyfour_copy_lss/masks'))

newd <- d

if (mack==1) {
  newd$sub <- factor(newd$sub, labels = c("I", "II", "III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX"))
} else {
  newd$sub <- factor(newd$sub, labels = c("I", "II", "III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV"))
}

#
fm1 <- lmer(corr ~ dist + sub + acc + roi + (roi|sub) + (dist|roi),
            REML=FALSE,
            nAGQ=0,
            lmerControl(optimizer = "nloptwrap", calc.derivs = FALSE),
            data=newd)

#optimizer = ""bobyqa


contrast.matrix <- rbind(
  `Pearson vs. Chebyshev` = c(0, -1, 0, 0, 0, 0, 0, 0,0,0,0,0,1,0,0,0,0,0,0,0
                              ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                              ,0,0,0,0,0),
  
  `Pearson vs. Cityblock` = c(0, 0, -1, 0, 0, 0, 0, 0,0,0,0,0,1,0,0,0,0,0,0,0
                              ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                              ,0,0,0,0,0),
  
  `Pearson vs. Cosine` = c(0, 0, 0, -1, 0, 0, 0, 0,0,0,0,0,1,0,0,0,0,0,0,0
                              ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                              ,0,0,0,0,0),
  
  `Pearson vs. Euclidean` = c(0, 0, 0, 0, -1, 0, 0, 0,0,0,0,0,1,0,0,0,0,0,0,0
                                  ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                  ,0,0,0,0,0),
  
  `Pearson vs. Inner product` = c(0, 0, 0, 0, 0, -1, 0, 0,0,0,0,0,1,0,0,0,0,0,0,0
                              ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                              ,0,0,0,0,0),
  
  `Pearson vs. Mahab diag` = c(0, 0, 0, 0, 0, 0, -1, 0,0,0,0,0,1,0,0,0,0,0,0,0
                                  ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                  ,0,0,0,0,0),
  
  `Pearson vs. Mahab no reg` = c(0, 0, 0, 0, 0, 0, 0, -1,0,0,0,0,1,0,0,0,0,0,0,0
                               ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                               ,0,0,0,0,0),
  
  `Pearson vs. Mahab` = c(0, 0, 0, 0, 0, 0, 0, 0,-1,0,0,0,1,0,0,0,0,0,0,0
                                 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                 ,0,0,0,0,0),
  
  `Pearson vs. Minkowski 10` = c(0, 0, 0, 0, 0, 0, 0, 0,0,-1,0,0,1,0,0,0,0,0,0,0
                          ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                          ,0,0,0,0,0),
  
  `Pearson vs. Minkowski 5` = c(0, 0, 0, 0, 0, 0, 0, 0,0,0,-1,0,1,0,0,0,0,0,0,0
                                 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                 ,0,0,0,0,0),
  
  `Pearson vs. Minkowski 50` = c(0, 0, 0, 0, 0, 0, 0, 0,0,0,0,-1,1,0,0,0,0,0,0,0
                                ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                ,0,0,0,0,0),
  
  
  `Pearson vs. Spearman` = c(0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,1,-1,0,0,0,0,0,0
                              ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                              ,0,0,0,0,0),
  
  `Pearson vs. Bhat` = c(0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,1,0,0,0,0,0,0,0
                             ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                             ,0,0,0,0,0),
  
  `Accuracy` = c(0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0
                         ,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
                         ,0,0,0,0,0)
  )


library(multcomp)
comps <- glht(fm1, contrast.matrix)
#summary(comps)
summary(glht(fm1, contrast.matrix), test = adjusted("bonferroni"))

cat("\n")
cat("\n dist")
cat("\n=======================\n")
fm2 <- update(fm1, .~. - dist - (dist|roi) )
chitest <- anova(fm1, fm2, test="Chisq")
cat("slope intercept ") 
cat(fixef(fm1)[1])
cat("\n")
cat("slope beta ") 
cat(fixef(fm1)[2])
cat("\n")
cat("chi-square ")
cat(chitest$Chisq[2])
cat("\n")
cat("beta p-value ")
cat(chitest$Pr[2])

cat("\n")
cat("\n roi")
cat("\n=======================\n")
fm3 <- update(fm1, .~. - roi - (roi|sub) )
chitest <- anova(fm1, fm3, test="Chisq")
cat("chi-square ")
cat(chitest$Chisq[2])
cat("\n")
cat("beta p-value ")
cat(chitest$Pr[2])

cat("\n")
cat("\n dist x roi")
cat("\n=======================\n")
fm4 <- update(fm1, .~. - (dist|roi) )
chitest <- anova(fm1, fm4, test="Chisq")
cat("chi-square ")
cat(chitest$Chisq[2])
cat("\n")
cat("beta p-value ")
cat(chitest$Pr[2])

#relgrad <- with(fm1@optinfo$derivs,solve(Hessian,gradient))
#max(abs(relgrad))