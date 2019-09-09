### XGBoost en Titanic data
library(tidyverse)
library(data.table)
library(magrittr)
library(ggridges)
library(xgboost)
library(rpart)
library(gridExtra)
library(vtreat)

train <- fread('C:/Users/hp/Documents/GitHub/statopia/archives/titanic_train.csv')
test <- fread('C:/Users/hp/Documents/GitHub/statopia/archives/titanic_test.csv')

# Combine both datasets 
test$Survived <- NA
train2 <- rbind(train, test)

# Simplifying the titles to reduce noise
train2$Title <- sapply(train2$Name, function(x) {strsplit(x, split = '[,.]')[[1]][[2]]})
train2$Title <- trimws(train2$Title, 'left')
train2$Title[train2$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
train2$Title[train2$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
train2$Title[train2$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
train2$Title <- factor(train2$Title)

# Creating a family size variable
train2 %<>% 
  mutate(family_size = SibSp + Parch + 1) 

# Imputing values to the missing values
fit_age <- rpart(Age ~ Sex + Pclass + Fare + Embarked + Title, data = train2)
train2$Age[is.na(train2$Age)] <- predict(fit_age, newdata = train2[is.na(train2$Age),])

# Fill gaps in Embarked and Fare
train2$Embarked <- ifelse(train2$Embarked == '', 'S', train2$Embarked) %>% as.factor()
fare_fit <- rpart(Fare ~ Age + Sex + Pclass + Embarked + Title, data = train2)
train2$Fare[is.na(train2$Fare)] <- predict(fare_fit, newdata = train2[is.na(train2$Fare),])

# All the missing values are now filled and the dataset is ready to be trained
summary(train2)

# Pick just the interesting columns
train3 <- train2[, c(2, 3, 5, 6, 7, 8, 10, 12, 13, 14)]

# Divide back to train/test split
train_m <- train3[!is.na(train3$Survived),]
test_m <- train3[is.na(train3$Survived),]
# Let's predict survivors 
# Surviving Rate By Sex

# train_plot$Survived <- as.factor(train_plot$Survived)
train_m %>% 
  group_by(Sex) %>% 
  summarise(total = n(),
            survived = sum(Survived, na.rm = T),
            non_survived = sum(ifelse(Survived == 1, 0, 1), na.rm = T)) %>%
  gather(key, num, survived, non_survived) %>% 
  mutate(prop = num / total) %>% 
  ggplot(aes(x = Sex, y = num, fill = key)) +
    geom_col(position='fill') +
    geom_label(aes(label = paste(num, '-', round(prop * 100, 2), '%')),
                   position = 'fill') +
    labs(x = 'Sex', y = 'Proportion', fill = '') +
    theme_minimal() +
    theme(axis.text = element_text(size = 12),
          plot.title = element_text(size = 20),
          legend.text = element_text(size = 12)) +
    scale_y_continuous(breaks = c(0, .25, .5, .75, 1),
                       labels = c('0%', '25%', '50%', '75%', '100%')) +
    scale_fill_discrete(labels = c("Non Survived", "Survived")) +
    ggtitle('Surviving Rate by Sex')

# Surviving by Fare
train_m %>% 
  filter(Fare < 300) %>% 
  ggplot(aes(x = Fare)) +
    geom_density(aes(fill = factor(Survived)), alpha = .5, size = 1.2) +
    theme_bw() +
    theme(axis.text = element_text(size = 12),
          plot.title = element_text(size = 20),
          legend.title = element_blank(),
          legend.text = element_text(size = 12)) +
    labs(x = 'Fare', y = 'Density') +
    scale_fill_discrete(labels = c("Non Survived", "Survived")) +
    ggtitle('Surviving Rate by Fare')

# Matrix creation

# designTreatmentsC for classifying models
#treatmentsN <- designTreatmentsC(train3, colnames(train3), 'Survived', outcometarget = '1') 
#feat_train <- prepare(treatmentsN, train3, pruneSig = 0.99) %>% as.matrix()
#response_train <- train3[,1] %>% as.matrix()

# And the same again for the test sample
# treatmentsN

# 4. matrix creation
# train_matrix <- xgb.DMatrix(train3)

### Model setting 
set.seed(123)

# 3. Creating matrix with the xgb matrix
#xgb.fit <- xgb.cv(
#  data = data.matrix(train3[,-1]),
#  label = train3[,1],
#  nrounds = 1000,
#  nfold = 5,
#  objective = 'binary:logistic',
#  verbose = 0
  # early_stopping_rounds = 10
#)

# 4. Model tuning
hyper_grid <- expand.grid(
  eta = c(.05, .1, .2),
  gamma = c(1, 1.2, 1.5),
  max_depth = c(3, 5, 7),
  min_child_weight = c(1, 3, 5),
  subsample = c(.5, .65, .8),
  colsample_bytree = c(.7, .8, .9),
  #optimal_trees = 0, 
  test_error_mean = 0
)
nrow(hyper_grid) # 729 possible combinations

# Model in a For loop
for (i in 1:nrow(hyper_grid)) {
  params <- list(
    eta = hyper_grid$eta[i],
    gamma = hyper_grid$gamma[i], 
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  xgb.tune <- xgb.cv(
    params = params,
    data = data.matrix(train_m[,-1]),
    label = train_m[,1],
    nrounds = 500,
    nfold = 5,
    objective = 'binary:logistic',
    #verbose = 0,
    early_stopping_rounds = 10
  )
  # add min training error and trees to grid
  #hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$test_error_mean[i] <- min(xgb.tune$evaluation_log$test_error_mean)
}

# Results in xgb.tune$evaluation_log!
hyper_grid %>%
  arrange(test_error_mean) %>%
  head(10)

hyper_grid %>%
  arrange(test_error_mean) %>%
  tail(10)

# Definitive model
params_final <- list(
  eta = 0.3,
  gamma = 1.2, 
  max_depth = 3, 
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = .9
)

xgb.fit <- xgb.cv(
  params = params_final,
  data = data.matrix(train3[,-1]),
  label = train3[,1],
  nrounds = 500,
  nfold = 5,
  objective = 'binary:logistic',
  #verbose = 0,
  early_stopping_rounds = 10,
  print_every_n = 100
)

xgb.fit <- xgboost(
  params = params_final,
  data = data.matrix(train3[,-1]),
  label = train3[,1],
  nrounds = 500,
  nfold = 5,
  objective = 'binary:logistic',
  #verbose = 0,
  early_stopping_rounds = 10,
  print_every_n = 100
)
# Test error: 0.24 


### PLOT
x1s <- seq(2, 5, length.out = 100)
x2s <- seq(1, 3, length.out = 100)
g <- data.frame(x1 = rep(x1s, each=100), x2 = rep(x2s, time = 100))
p <- predict(xgb.fit,newdata=data.matrix(g[,c('x1','x2')]))
g$y <- ifelse(p>=0.90,1,0)
g1 <- ggplot(data=dataTest[dataTest$x1>2 & dataTest$x1<5 & dataTest$x2>1 & dataTest$x2<3,]) +
  xlim(2,5) + ylim(1,3) +
  geom_tile(data=g,aes(x1,x2,fill=factor(y))) +
  geom_point(size=2,aes(x1,x2,color=factor(p),shape=factor(Pclass))) +
  scale_color_manual(values=c('#666666','#0000FF'),
                     limits=c('0','1'),labels=c('0','1')) +
  scale_fill_manual(values=c('#FF9999','#99FF99'),
                    limits=c('0','1'),labels=c('0','1')) +
  labs(x='Fare / (10 x TicketFrequency)',y='FamilySize + (Age / 70)',shape='Pclass',fill='Classifier',
       title='XGBoost classifies the test set.
        It predicts 4 adult males have P(live)>=0.9',color='Predict') +
  geom_vline(xintercept=2.8, linetype='dotted') +
  geom_hline(yintercept=c(1.43,2.43), linetype='dotted') +
  annotate('text',x=2.95,y=2.9,label='Fare = $28') +
  annotate('text',x=4.7,y=2.35,label='Age = 30') +
  annotate('text',x=4.7,y=1.35,label='Age = 30')

for (i in which(dataTest$p==1)){
  g1 <- g1 + annotate('text',x=dataTest$x1[i]-0.15,y=dataTest$x2[i],label=dataTest$PassengerId[i]
                      ,color='darkblue',size=4)
}
g1


####### CODE SNIPPET 2

y = dtrain[,1]

train = as.matrix(dtrain[,-1])

test = as.matrix(dtest)

GS_LogLoss = data.frame("Rounds" = numeric(), 
                        "Depth" = numeric(),
                        "r_sample" = numeric(),
                        "c_sample" = numeric(), 
                        "minLogLoss" = numeric(),
                        "best_round" = numeric())

for (rounds in seq(100, 1000, 50)){
  
  for (depth in c(4, 6, 8, 10)) {
    
    for (r_sample in c(0.5, 0.75, 1)) {
      
      for (c_sample in c(0.4, 0.6, 0.8, 1)) {
        
        set.seed(1024)
        eta_val = 2 / rounds
        cv.res = xgb.cv(data = train, nfold = 3, label = y, 
                        nrounds = rounds, 
                        eta = eta_val, 
                        max_depth = depth,
                        subsample = r_sample, 
                        colsample_bytree = c_sample,
                        early.stop.round = 0.5*rounds,
                        objective='binary:logistic', 
                        eval_metric = 'logloss',
                        verbose = FALSE)
        
        print(paste(rounds, depth, r_sample, c_sample, min(as.matrix(cv.res)[,3]) ))
        GS_LogLoss[nrow(GS_LogLoss)+1, ] = c(rounds, 
                                             depth, 
                                             r_sample, 
                                             c_sample, 
                                             min(as.matrix(cv.res)[,3]), 
                                             which.min(as.matrix(cv.res)[,3]))
        
      }
    }
  }
}

#######