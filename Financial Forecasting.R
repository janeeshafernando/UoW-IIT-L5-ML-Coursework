# Loading the necessary libraries 
library(dplyr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(neuralnet)
library(grid)
library(MASS)
library(Metrics)
library(MLmetrics)

# Load the dataset
exchange_rates <- read.csv("ExchangeUSD.csv",header = TRUE)
str(exchange_rates)
summary(exchange_rates)

# Create a new data frame exchange_rate_time_series that contains only the USD.EUR column.
exchange_rate_time_series <- as.data.frame(exchange_rates[,3])

# Function to scale the data (min-max normalization)
normalize <- function(x){
  return( (x-min(x)) / (max(x) - min(x)))
}

# Create the reverse of normalized function - de-normalized
unnormalize <- function(x, min, max){
  return((max - min)*x + min)
}

###################################################################################################

# I/O Matrix (T-1)

###################################################################################################

t1_time_lagged_data <- bind_cols(G_current = lag(exchange_rate_time_series, 1),
                                      G_pred = exchange_rate_time_series)

colnames(t1_time_lagged_data) <- c("Input", "Output")

# Remove the rows with NA
t1_time_lagged_data <- t1_time_lagged_data[complete.cases(t1_time_lagged_data), ]
View(t1_time_lagged_data)

#Store the last index of the timed_lagged_data
t1_time_lagged_last_index <- nrow(t1_time_lagged_data)
t1_time_lagged_last_index

#Scale the I/O Matrix (min-max normalization)
t1_normalise_data <- as.data.frame(lapply(t1_time_lagged_data,normalize))
head(t1_normalise_data)
summary(t1_normalise_data)

#Split the training and testing dataset
t1_train_data <- t1_normalise_data[1:400,]
t1_test_data <- t1_normalise_data[401:t1_time_lagged_last_index,]

#################################################################################################################################

# MLP models for Case T-1

#################################################################################################################################

# MLP Model 1

t1_model_1 <- neuralnet(Output ~ Input, data = t1_train_data, hidden = 6, act.fct = "logistic",
                        err.fct = 'sse',linear.output = T)

plot(t1_model_1)

# Evaluating the model
# Generate predictions on the test data
t1_model_results_1 <- compute(t1_model_1, t1_test_data[, 1, drop = FALSE])

# Obtain predicted result
t1_predicted_results_1 <- t1_model_results_1$net.result
t1_predicted_results_1

t1_time_lagged_train <- t1_time_lagged_data[1:400,"Output"]
summary(t1_time_lagged_train)

t1_time_lagged_test <- t1_time_lagged_data[401:t1_time_lagged_last_index,"Output"]
summary(t1_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t1_min <- min(t1_time_lagged_train)
train_t1_max <- max(t1_time_lagged_train)

train_t1_max
train_t1_min

# de-normalize the normalized NN's output
t1_model_pred_1 <- unnormalize(t1_predicted_results_1,train_t1_min,train_t1_max)
head(t1_model_pred_1)

#Calculate the Root Mean Squared Error(RMSE)
t1_rmse_1 <- rmse(t1_time_lagged_test,t1_model_pred_1)
t1_rmse_1

# Mean Absolute Error (MAE)
t1_mae_1 <- mae(t1_time_lagged_test,t1_model_pred_1)
t1_mae_1

# Mean Absolute Percentage Error (MAPE)
t1_mape_1 <- mape(t1_time_lagged_test,t1_model_pred_1)
t1_mape_1

# Symmetric Mean Absolute Percentage Error (sMAPE)
t1_smape_1 <- smape(t1_time_lagged_test,t1_model_pred_1)
t1_smape_1

t1_clean_output_1 <- cbind(t1_time_lagged_test,t1_model_pred_1)
colnames(t1_clean_output_1) <- c("Output","Result")
View(t1_clean_output_1)

# Visual Plot
par(mfrow=c(1,1))
plot(t1_time_lagged_test,t1_model_pred_1 ,col='red',main="Real vs Predicted Exchange Rate",pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t1_time_lagged_test)
plot(x, t1_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t1_model_pred_1, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

############################################################################################################################

# MLP Model 2

t1_model_2 <- neuralnet(Output ~ Input, data = t1_train_data, hidden = c(8,4), act.fct = "tanh",
                        err.fct = 'sse',linear.output = F)

plot(t1_model_2)

# Evaluating the model
# Generate predictions on the test data
t1_model_results_2 <- compute(t1_model_2, t1_test_data[, 1, drop = FALSE])

# Obtain predicted result
t1_predicted_results_2 <- t1_model_results_2$net.result
t1_predicted_results_2

t1_time_lagged_train <- t1_time_lagged_data[1:400,"Output"]
summary(t1_time_lagged_train)

t1_time_lagged_test <- t1_time_lagged_data[401:t1_time_lagged_last_index,"Output"]
summary(t1_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t1_min <- min(t1_time_lagged_train)
train_t1_max <- max(t1_time_lagged_train)

train_t1_max
train_t1_min

# de-normalize the normalized NN's output
t1_model_pred_2 <- unnormalize(t1_predicted_results_2,train_t1_min,train_t1_max)
head(t1_model_pred_2)

#Calculate the Root Mean Squared Error(RMSE)
t1_rmse_2 <- rmse(t1_time_lagged_test,t1_model_pred_2)
t1_rmse_2

# Mean Absolute Error (MAE)
t1_mae_2 <- mae(t1_time_lagged_test,t1_model_pred_2)
t1_mae_2

# Mean Absolute Percentage Error (MAPE)
t1_mape_2 <- mape(t1_time_lagged_test,t1_model_pred_2)
t1_mape_2

# Symmetric Mean Absolute Percentage Error (sMAPE)
t1_smape_2 <- smape(t1_time_lagged_test,t1_model_pred_2)
t1_smape_2

t1_clean_output_2 <- cbind(t1_time_lagged_test,t1_model_pred_2)
colnames(t1_clean_output_2) <- c("Output","Result")
t1_clean_output_2

# Visual Plot
par(mfrow=c(1,1))
plot(t1_time_lagged_test,t1_model_pred_2,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t1_time_lagged_test)
plot(x, t1_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t1_model_pred_2, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

#############################################################################################################

# MLP Model 3

t1_model_3 <- neuralnet(Output ~ Input, data = t1_train_data, hidden = c(3,5,10), act.fct = "logistic",
                        err.fct = 'sse',linear.output = F)

plot(t1_model_3)

# Evaluating the model
# Generate predictions on the test data
t1_model_results_3 <- compute(t1_model_3, t1_test_data[, 1, drop = FALSE])

# Obtain predicted result
t1_predicted_results_3 <- t1_model_results_3$net.result
t1_predicted_results_3

t1_time_lagged_train <- t1_time_lagged_data[1:400,"Output"]
summary(t1_time_lagged_train)

t1_time_lagged_test <- t1_time_lagged_data[401:t1_time_lagged_last_index,"Output"]
summary(t1_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t1_min <- min(t1_time_lagged_train)
train_t1_max <- max(t1_time_lagged_train)

train_t1_max
train_t1_min

# de-normalize the normalized NN's output
t1_model_pred_3 <- unnormalize(t1_predicted_results_3,train_t1_min,train_t1_max)
head(t1_model_pred_3)

#Calculate the Root Mean Squared Error(RMSE)
t1_rmse_3 <- rmse(t1_time_lagged_test,t1_model_pred_3)
t1_rmse_3

# Mean Absolute Error (MAE)
t1_mae_3 <- mae(t1_time_lagged_test,t1_model_pred_3)
t1_mae_3

# Mean Absolute Percentage Error (MAPE)
t1_mape_3 <- mape(t1_time_lagged_test,t1_model_pred_3)
t1_mape_3

# Symmetric Mean Absolute Percentage Error (sMAPE)
t1_smape_3 <- smape(t1_time_lagged_test,t1_model_pred_3)
t1_smape_3

t1_clean_output_3 <- cbind(t1_time_lagged_test,t1_model_pred_3)
colnames(t1_clean_output_3) <- c("Output","Result")
t1_clean_output_3

# Visual Plot
par(mfrow=c(1,1))
plot(t1_time_lagged_test,t1_model_pred_3,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t1_time_lagged_test)
plot(x, t1_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t1_model_pred_3, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 


####################################################################################################

# I/O Matrix (T-2)

####################################################################################################

t2_time_lagged_data <- bind_cols(G_previous = lag(exchange_rate_time_series, 2),
                                 G_current = lag(exchange_rate_time_series, 1),
                                 G_pred = exchange_rate_time_series)

colnames(t2_time_lagged_data) <- c("Input1","Input2", "Output")

# Remove the rows with NA
t2_time_lagged_data <- t2_time_lagged_data[complete.cases(t2_time_lagged_data), ]

View(t2_time_lagged_data)

#Store the last index of the timed_lagged_data
t2_time_lagged_last_index <- nrow(t2_time_lagged_data)
t2_time_lagged_last_index

#Scale the I/O Matrix (min-max normalization)
t2_normalise_data <- as.data.frame(lapply(t2_time_lagged_data,normalize))
head(t2_normalise_data)
summary(t2_normalise_data)

#Split the training and testing dataset
t2_train_data <- t2_normalise_data[1:400,]
t2_test_data <- t2_normalise_data[401:t2_time_lagged_last_index,]

#################################################################################################################################

# MLP models for Case T-2

#################################################################################################################################

# MLP Model 1

t2_model_1 <- neuralnet(Output ~ Input1 + Input2, data = t2_train_data, hidden = 8, 
                        act.fct = "logistic",err.fct = 'sse',linear.output = T)

plot(t2_model_1)

# Evaluating the model
# Generate predictions on the test data
t2_model_results_1 <- compute(t2_model_1,t2_test_data[1:2])

# Obtain predicted result
t2_predicted_results_1 <- t2_model_results_1$net.result
t2_predicted_results_1

t2_time_lagged_train <- t2_time_lagged_data[1:400,"Output"]
summary(t2_time_lagged_train)

t2_time_lagged_test <- t2_time_lagged_data[401:t2_time_lagged_last_index,"Output"]
summary(t2_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t2_min <- min(t2_time_lagged_train)
train_t2_max <- max(t2_time_lagged_train)

train_t2_max
train_t2_min

# de-normalize the normalized NN's output
t2_model_pred_1 <- unnormalize(t2_predicted_results_1,train_t2_min,train_t2_max)
head(t2_model_pred_1)

#Calculate the Root Mean Squared Error(RMSE)
t2_rmse_1 <- rmse(t2_time_lagged_test,t2_model_pred_1)
t2_rmse_1

# Mean Absolute Error (MAE)
t2_mae_1 <- mae(t2_time_lagged_test,t2_model_pred_1)
t2_mae_1

# Mean Absolute Percentage Error (MAPE)
t2_mape_1 <- mape(t2_time_lagged_test,t2_model_pred_1)
t2_mape_1

# Symmetric Mean Absolute Percentage Error (sMAPE)
t2_smape_1 <- smape(t2_time_lagged_test,t2_model_pred_1)
t2_smape_1

t2_clean_output_1 <- cbind(t2_time_lagged_test,t2_model_pred_1)
colnames(t2_clean_output_1) <- c("Output","Result")
t2_clean_output_1

# Visual Plot
par(mfrow=c(1,1))
plot(t2_time_lagged_test,t2_model_pred_1 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t2_time_lagged_test)
plot(x, t2_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t2_model_pred_1, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

###########################################################################################################################

# MLP Model 2

t2_model_2 <- neuralnet(Output ~ Input1 + Input2, data = t2_train_data, hidden = c(4,10), 
                        act.fct = "logistic",err.fct = 'sse',linear.output = T)

plot(t2_model_2)

# Evaluating the model
# Generate predictions on the test data
t2_model_results_2 <- compute(t2_model_2,t2_test_data[1:2])

# Obtain predicted result
t2_predicted_results_2 <- t2_model_results_2$net.result
t2_predicted_results_2

t2_time_lagged_train <- t2_time_lagged_data[1:400,"Output"]
summary(t2_time_lagged_train)

t2_time_lagged_test <- t2_time_lagged_data[401:t2_time_lagged_last_index,"Output"]
summary(t2_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t2_min <- min(t2_time_lagged_train)
train_t2_max <- max(t2_time_lagged_train)

train_t2_max
train_t2_min

# de-normalize the normalized NN's output
t2_model_pred_2 <- unnormalize(t2_predicted_results_2,train_t2_min,train_t2_max)
head(t2_model_pred_2)

#Calculate the Root Mean Squared Error(RMSE)
t2_rmse_2 <- rmse(t2_time_lagged_test,t2_model_pred_2)
t2_rmse_2

# Mean Absolute Error (MAE)
t2_mae_2 <- mae(t2_time_lagged_test,t2_model_pred_2)
t2_mae_2

# Mean Absolute Percentage Error (MAPE)
t2_mape_2 <- mape(t2_time_lagged_test,t2_model_pred_2)
t2_mape_2

# Symmetric Mean Absolute Percentage Error (sMAPE)
t2_smape_2 <- smape(t2_time_lagged_test,t2_model_pred_2)
t2_smape_2

t2_clean_output_2 <- cbind(t2_time_lagged_test,t2_model_pred_2)
colnames(t2_clean_output_2) <- c("Output","Result")
t2_clean_output_2

# Visual Plot
par(mfrow=c(1,1))
plot(t2_time_lagged_test,t2_model_pred_2 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t2_time_lagged_test)
plot(x, t2_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t2_model_pred_2, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

###########################################################################################################################

# MLP Model 2

t2_model_2 <- neuralnet(Output ~ Input1 + Input2, data = t2_train_data, hidden = c(6,10), act.fct = "logistic",err.fct = 'sse',linear.output = T)

plot(t2_model_2)

# Evaluating the model
# Generate predictions on the test data
t2_model_results_2 <- compute(t2_model_2,t2_test_data[1:2])

# Obtain predicted result
t2_predicted_results_2 <- t2_model_results_2$net.result
t2_predicted_results_2

t2_time_lagged_train <- t2_time_lagged_data[1:400,"Output"]
summary(t2_time_lagged_train)

t2_time_lagged_test <- t2_time_lagged_data[401:t2_time_lagged_last_index,"Output"]
summary(t2_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t2_min <- min(t2_time_lagged_train)
train_t2_max <- max(t2_time_lagged_train)

train_t2_max
train_t2_min

# de-normalize the normalized NN's output
t2_model_pred_2 <- unnormalize(t2_predicted_results_2,train_t2_min,train_t2_max)
head(t2_model_pred_2)

#Calculate the Root Mean Squared Error(RMSE)
t2_rmse_2 <- rmse(t2_time_lagged_test,t2_model_pred_2)
t2_rmse_2

# Mean Absolute Error (MAE)
t2_mae_2 <- mae(t2_time_lagged_test,t2_model_pred_2)
t2_mae_2

# Mean Absolute Percentage Error (MAPE)
t2_mape_2 <- mape(t2_time_lagged_test,t2_model_pred_2)
t2_mape_2

# Symmetric Mean Absolute Percentage Error (sMAPE)
t2_smape_2 <- smape(t2_time_lagged_test,t2_model_pred_2)
t2_smape_2

t2_clean_output_2 <- cbind(t2_time_lagged_test,t2_model_pred_2)
colnames(t2_clean_output_2) <- c("Output","Result")
t2_clean_output_2

# Visual Plot
par(mfrow=c(1,1))
plot(t2_time_lagged_test,t2_model_pred_2 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t2_time_lagged_test)
plot(x, t2_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t2_model_pred_2, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 


# MLP Model 3

t2_model_3 <- neuralnet(Output ~ Input1 + Input2, data = t2_train_data, hidden = c(4,7,16), 
                        act.fct = "tanh",err.fct = 'sse',linear.output = F)

plot(t2_model_3)

# Evaluating the model
# Generate predictions on the test data
t2_model_results_3 <- compute(t2_model_3,t2_test_data[1:2])

# Obtain predicted result
t2_predicted_results_3 <- t2_model_results_3$net.result
t2_predicted_results_3

t2_time_lagged_train <- t2_time_lagged_data[1:400,"Output"]
summary(t2_time_lagged_train)

t2_time_lagged_test <- t2_time_lagged_data[401:t2_time_lagged_last_index,"Output"]
summary(t2_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t2_min <- min(t2_time_lagged_train)
train_t2_max <- max(t2_time_lagged_train)

train_t2_max
train_t2_min

# de-normalize the normalized NN's output
t2_model_pred_3 <- unnormalize(t2_predicted_results_3,train_t2_min,train_t2_max)
head(t2_model_pred_3)

#Calculate the Root Mean Squared Error(RMSE)
t2_rmse_3 <- rmse(t2_time_lagged_test,t2_model_pred_3)
t2_rmse_3

# Mean Absolute Error (MAE)
t2_mae_3 <- mae(t2_time_lagged_test,t2_model_pred_3)
t2_mae_3

# Mean Absolute Percentage Error (MAPE)
t2_mape_3 <- mape(t2_time_lagged_test,t2_model_pred_3)
t2_mape_3

# Symmetric Mean Absolute Percentage Error (sMAPE)
t2_smape_3 <- smape(t2_time_lagged_test,t2_model_pred_3)
t2_smape_3

t2_clean_output_3 <- cbind(t2_time_lagged_test,t2_model_pred_3)
colnames(t2_clean_output_3) <- c("Output","Result")
t2_clean_output_3

# Visual Plot
par(mfrow=c(1,1))
plot(t2_time_lagged_test,t2_model_pred_3 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t2_time_lagged_test)
plot(x, t2_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t2_model_pred_3, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

############################################################################################################################

# I/O Matrix (T-3)

############################################################################################################################

t3_time_lagged_data <- bind_cols(G_previous2 = lag(exchange_rate_time_series, 3),
                                G_previous1 = lag(exchange_rate_time_series, 2),
                                G_current = lag(exchange_rate_time_series, 1),
                                G_pred = exchange_rate_time_series)

colnames(t3_time_lagged_data) <- c("Input1","Input2","Input3","Output")

# Remove the rows with NA
t3_time_lagged_data <- t3_time_lagged_data[complete.cases(t3_time_lagged_data), ]

View(t3_time_lagged_data)

#Store the last index of the timed_lagged_data
t3_time_lagged_last_index <- nrow(t3_time_lagged_data)
t3_time_lagged_last_index

#Scale the I/O Matrix (min-max normalization)
t3_normalise_data <- as.data.frame(lapply(t3_time_lagged_data,normalize))
head(t3_normalise_data)
summary(t3_normalise_data)

#Split the training and testing dataset
t3_train_data <- t3_normalise_data[1:400,]
t3_test_data <- t3_normalise_data[401:t3_time_lagged_last_index,]


#################################################################################################################################

# MLP models for Case T-3

#################################################################################################################################

# MLP Model 1

t3_model_1 <- neuralnet(Output ~ Input1 + Input2 + Input3, data = t3_train_data, hidden = 10, 
                        act.fct = "logistic",err.fct = 'sse',linear.output = F)

plot(t3_model_1)

# Evaluating the model
# Generate predictions on the test data
t3_model_results_1 <- compute(t3_model_1,t3_test_data[1:3])

# Obtain predicted result
t3_predicted_results_1 <- t3_model_results_1$net.result
t3_predicted_results_1

t3_time_lagged_train <- t3_time_lagged_data[1:400,"Output"]
summary(t3_time_lagged_train)

t3_time_lagged_test <- t3_time_lagged_data[401:t3_time_lagged_last_index,"Output"]
summary(t3_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t3_min <- min(t3_time_lagged_train)
train_t3_max <- max(t3_time_lagged_train)

train_t3_max
train_t3_min

# de-normalize the normalized NN's output
t3_model_pred_1 <- unnormalize(t3_predicted_results_1,train_t3_min,train_t3_max)
head(t3_model_pred_1)

#Calculate the Root Mean Squared Error(RMSE)
t3_rmse_1 <- rmse(t3_time_lagged_test,t3_model_pred_1)
t3_rmse_1

# Mean Absolute Error (MAE)
t3_mae_1 <- mae(t3_time_lagged_test,t3_model_pred_1)
t3_mae_1

# Mean Absolute Percentage Error (MAPE)
t3_mape_1 <- mape(t3_time_lagged_test,t3_model_pred_1)
t3_mape_1

# Symmetric Mean Absolute Percentage Error (sMAPE)
t3_smape_1 <- smape(t3_time_lagged_test,t3_model_pred_1)
t3_smape_1

t3_clean_output_1 <- cbind(t3_time_lagged_test,t3_model_pred_1)
colnames(t3_clean_output_1) <- c("Output","Result")
t3_clean_output_1

# Visual Plot
par(mfrow=c(1,1))
plot(t3_time_lagged_test,t3_model_pred_1 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t3_time_lagged_test)
plot(x, t3_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t3_model_pred_1, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

###################################################################################################

# MLP Model 2

t3_model_2 <- neuralnet(Output ~ Input1 + Input2 + Input3, data = t3_train_data, hidden = c(8,16),
                        act.fct = "tanh",err.fct = 'sse',linear.output = T)

plot(t3_model_2)

# Evaluating the model
# Generate predictions on the test data
t3_model_results_2 <- compute(t3_model_2,t3_test_data[1:3])

# Obtain predicted result
t3_predicted_results_2 <- t3_model_results_2$net.result
t3_predicted_results_2

t3_time_lagged_train <- t3_time_lagged_data[1:400,"Output"]
summary(t3_time_lagged_train)

t3_time_lagged_test <- t3_time_lagged_data[401:t3_time_lagged_last_index,"Output"]
summary(t3_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t3_min <- min(t3_time_lagged_train)
train_t3_max <- max(t3_time_lagged_train)

train_t3_max
train_t3_min

# de-normalize the normalized NN's output
t3_model_pred_2 <- unnormalize(t3_predicted_results_2,train_t3_min,train_t3_max)
head(t3_model_pred_2)

#Calculate the Root Mean Squared Error(RMSE)
t3_rmse_2 <- rmse(t3_time_lagged_test,t3_model_pred_2)
t3_rmse_2

# Mean Absolute Error (MAE)
t3_mae_2 <- mae(t3_time_lagged_test,t3_model_pred_2)
t3_mae_2

# Mean Absolute Percentage Error (MAPE)
t3_mape_2 <- mape(t3_time_lagged_test,t3_model_pred_2)
t3_mape_2

# Symmetric Mean Absolute Percentage Error (sMAPE)
t3_smape_2 <- smape(t3_time_lagged_test,t3_model_pred_2)
t3_smape_2

t3_clean_output_2 <- cbind(t3_time_lagged_test,t3_model_pred_2)
colnames(t3_clean_output_2) <- c("Output","Result")
t3_clean_output_2

# Visual Plot
par(mfrow=c(1,1))
plot(t3_time_lagged_test,t3_model_pred_2 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t3_time_lagged_test)
plot(x, t3_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t3_model_pred_2, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

########################################################################################################

# I/O Matrix (T-4)

########################################################################################################

t4_time_lagged_data <- bind_cols(G_previous3 = lag(exchange_rate_time_series, 4),
                                G_previous2 = lag(exchange_rate_time_series, 3),
                                G_previous1 = lag(exchange_rate_time_series, 2),
                                G_current = lag(exchange_rate_time_series, 1),
                                G_pred = exchange_rate_time_series)

colnames(t4_time_lagged_data) <- c("Input1","Input2","Input3","Input4","Output")

# Remove the rows with NA
t4_time_lagged_data <- t4_time_lagged_data[complete.cases(t4_time_lagged_data), ]

View(t4_time_lagged_data)

#Store the last index of the timed_lagged_data
t4_time_lagged_last_index <- nrow(t4_time_lagged_data)
t4_time_lagged_last_index

#Scale the I/O Matrix (min-max normalization)
t4_normalise_data <- as.data.frame(lapply(t4_time_lagged_data,normalize))
head(t4_normalise_data)
summary(t4_normalise_data)

#Split the training and testing dataset
t4_train_data <- t4_normalise_data[1:400,]
t4_test_data <- t4_normalise_data[401:t4_time_lagged_last_index,]

###################################################################################################

# MLP Model 1

t4_model_1 <- neuralnet(Output ~ Input1 + Input2 + Input3 + Input4, data = t4_train_data, hidden = 8,
                        act.fct = "tanh",err.fct = 'sse',linear.output = T)

plot(t4_model_1)

# Evaluating the model
# Generate predictions on the test data
t4_model_results_1 <- compute(t4_model_1,t4_test_data[1:4])

# Obtain predicted result
t4_predicted_results_1 <- t4_model_results_1$net.result
t4_predicted_results_1

t4_time_lagged_train <- t4_time_lagged_data[1:400,"Output"]
summary(t4_time_lagged_train)

t4_time_lagged_test <- t4_time_lagged_data[401:t4_time_lagged_last_index,"Output"]
summary(t4_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t4_min <- min(t4_time_lagged_train)
train_t4_max <- max(t4_time_lagged_train)

train_t4_max
train_t4_min

# de-normalize the normalized NN's output
t4_model_pred_1 <- unnormalize(t4_predicted_results_1,train_t4_min,train_t4_max)
head(t4_model_pred_1)

#Calculate the Root Mean Squared Error(RMSE)
t4_rmse_1 <- rmse(t4_time_lagged_test,t4_model_pred_1)
t4_rmse_1

# Mean Absolute Error (MAE)
t4_mae_1 <- mae(t4_time_lagged_test,t4_model_pred_1)
t4_mae_1

# Mean Absolute Percentage Error (MAPE)
t4_mape_1 <- mape(t4_time_lagged_test,t4_model_pred_1)
t4_mape_1

# Symmetric Mean Absolute Percentage Error (sMAPE)
t4_smape_1 <- smape(t4_time_lagged_test,t4_model_pred_1)
t4_smape_1

t4_clean_output_1 <- cbind(t4_time_lagged_test,t4_model_pred_1)
colnames(t4_clean_output_1) <- c("Output","Result")
t4_clean_output_1

# Visual Plot
par(mfrow=c(1,1))
plot(t4_time_lagged_test,t4_model_pred_1 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t4_time_lagged_test)
plot(x, t4_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t4_model_pred_1, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

###################################################################################################

# MLP Model 2

t4_model_2 <- neuralnet(Output ~ Input1 + Input2 + Input3 + Input4, data = t4_train_data, 
                        hidden = c(6,12), act.fct = "tanh",err.fct = 'sse',linear.output = T)

plot(t4_model_2)

# Evaluating the model
# Generate predictions on the test data
t4_model_results_2 <- compute(t4_model_2,t4_test_data[1:4])

# Obtain predicted result
t4_predicted_results_2 <- t4_model_results_2$net.result
t4_predicted_results_2

t4_time_lagged_train <- t4_time_lagged_data[1:400,"Output"]
summary(t4_time_lagged_train)

t4_time_lagged_test <- t4_time_lagged_data[401:t4_time_lagged_last_index,"Output"]
summary(t4_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t4_min <- min(t4_time_lagged_train)
train_t4_max <- max(t4_time_lagged_train)

train_t4_max
train_t4_min

# de-normalize the normalized NN's output
t4_model_pred_2 <- unnormalize(t4_predicted_results_2,train_t4_min,train_t4_max)
head(t4_model_pred_2)

#Calculate the Root Mean Squared Error(RMSE)
t4_rmse_2 <- rmse(t4_time_lagged_test,t4_model_pred_2)
t4_rmse_2

# Mean Absolute Error (MAE)
t4_mae_2 <- mae(t4_time_lagged_test,t4_model_pred_2)
t4_mae_2

# Mean Absolute Percentage Error (MAPE)
t4_mape_2 <- mape(t4_time_lagged_test,t4_model_pred_2)
t4_mape_2

# Symmetric Mean Absolute Percentage Error (sMAPE)
t4_smape_2 <- smape(t4_time_lagged_test,t4_model_pred_2)
t4_smape_2

t4_clean_output_2 <- cbind(t4_time_lagged_test,t4_model_pred_2)
colnames(t4_clean_output_2) <- c("Output","Result")
t4_clean_output_2

# Visual Plot
par(mfrow=c(1,1))
plot(t4_time_lagged_test,t4_model_pred_2 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t4_time_lagged_test)
plot(x, t4_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t4_model_pred_2, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 


###################################################################################################

# I/O Matrix (T-5)

###################################################################################################

t5_time_lagged_data <- bind_cols(G_previous4 = lag(exchange_rate_time_series, 5),
                                G_previous3 = lag(exchange_rate_time_series, 4),
                                G_previous2 = lag(exchange_rate_time_series, 3),
                                G_previous1 = lag(exchange_rate_time_series, 2),
                                G_current = lag(exchange_rate_time_series, 1),
                                G_pred = exchange_rate_time_series)

colnames(t5_time_lagged_data) <- c("Input1","Input2","Input3","Input4","Input5","Output")

# Remove the rows with NA
t5_time_lagged_data <- t5_time_lagged_data[complete.cases(t5_time_lagged_data), ]

View(t5_time_lagged_data)

#Scale the I/O Matrix (min-max normalization)
t5_normalise_data <- as.data.frame(lapply(t5_time_lagged_data,normalize))
head(t5_normalise_data)
summary(t5_normalise_data)

#Store the last index of the timed_lagged_data
t5_time_lagged_last_index <- nrow(t5_normalise_data)
t5_time_lagged_last_index

#Split the training and testing dataset
t5_train_data <- t5_normalise_data[1:400,]
t5_test_data <- t5_normalise_data[401:t5_time_lagged_last_index,]

###################################################################################################

# MLP Model 1

t5_model_1 <- neuralnet(Output ~ Input1 + Input2 + Input3 + Input4 + Input5, data = t5_train_data,
                        hidden = 10, act.fct = "logistic",err.fct = 'sse',linear.output = T)

plot(t5_model_1)

# Evaluating the model
# Generate predictions on the test data
t5_model_results_1 <- compute(t5_model_1,t5_test_data[1:5])

# Obtain predicted result
t5_predicted_results_1 <- t5_model_results_1$net.result
t5_predicted_results_1

t5_time_lagged_train <- t5_time_lagged_data[1:400,"Output"]
summary(t5_time_lagged_train)

t5_time_lagged_test <- t5_time_lagged_data[401:t5_time_lagged_last_index,"Output"]
summary(t5_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t5_min <- min(t5_time_lagged_train)
train_t5_max <- max(t5_time_lagged_train)

train_t5_max
train_t5_min

# de-normalize the normalized NN's output
t5_model_pred_1 <- unnormalize(t5_predicted_results_1,train_t5_min,train_t5_max)
head(t5_model_pred_1)

#Calculate the Root Mean Squared Error(RMSE)
t5_rmse_1 <- rmse(t5_time_lagged_test,t5_model_pred_1)
t5_rmse_1

# Mean Absolute Error (MAE)
t5_mae_1 <- mae(t5_time_lagged_test,t5_model_pred_1)
t5_mae_1

# Mean Absolute Percentage Error (MAPE)
t5_mape_1 <- mape(t5_time_lagged_test,t5_model_pred_1)
t5_mape_1

# Symmetric Mean Absolute Percentage Error (sMAPE)
t5_smape_1 <- smape(t5_time_lagged_test,t5_model_pred_1)
t5_smape_1

t5_clean_output_1 <- cbind(t5_time_lagged_test,t5_model_pred_1)
colnames(t5_clean_output_1) <- c("Output","Result")
t5_clean_output_1

# Visual Plot
par(mfrow=c(1,1))
plot(t5_time_lagged_test,t5_model_pred_1 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t5_time_lagged_test)
plot(x, t5_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t5_model_pred_1, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

#################################################################################################

# MLP Model 2

t5_model_2 <- neuralnet(Output ~ Input1 + Input2 + Input3 + Input4 + Input5, data = t5_train_data, 
                        hidden = c(7,16), act.fct = "tanh",err.fct = 'sse',linear.output = T)

plot(t5_model_2)

# Evaluating the model
# Generate predictions on the test data
t5_model_results_2 <- compute(t5_model_2,t5_test_data[1:5])

# Obtain predicted result
t5_predicted_results_2 <- t5_model_results_2$net.result
t5_predicted_results_2

t5_time_lagged_train <- t5_time_lagged_data[1:400,"Output"]
summary(t5_time_lagged_train)

t5_time_lagged_test <- t5_time_lagged_data[401:t5_time_lagged_last_index,"Output"]
summary(t5_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t5_min <- min(t5_time_lagged_train)
train_t5_max <- max(t5_time_lagged_train)

train_t5_max
train_t5_min

# de-normalize the normalized NN's output
t5_model_pred_2 <- unnormalize(t5_predicted_results_2,train_t5_min,train_t5_max)
head(t5_model_pred_2)

#Calculate the Root Mean Squared Error(RMSE)
t5_rmse_2 <- rmse(t5_time_lagged_test,t5_model_pred_2)
t5_rmse_2

# Mean Absolute Error (MAE)
t5_mae_2 <- mae(t5_time_lagged_test,t5_model_pred_2)
t5_mae_2

# Mean Absolute Percentage Error (MAPE)
t5_mape_2 <- mape(t5_time_lagged_test,t5_model_pred_2)
t5_mape_2

# Symmetric Mean Absolute Percentage Error (sMAPE)
t5_smape_2 <- smape(t5_time_lagged_test,t5_model_pred_2)
t5_smape_2

t5_clean_output_2 <- cbind(t5_time_lagged_test,t5_model_pred_2)
colnames(t5_clean_output_2) <- c("Output","Result")
t5_clean_output_2

# Visual Plot
par(mfrow=c(1,1))
plot(t5_time_lagged_test,t5_model_pred_2 ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t5_time_lagged_test)
plot(x, t5_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t5_model_pred_2, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

##################################################################################################

# I/O Matrix (T-6)

##################################################################################################

t6_time_lagged_data <- bind_cols(G_previous5 = lag(exchange_rate_time_series, 6),
                                G_previous4 = lag(exchange_rate_time_series, 5),
                                G_previous3 = lag(exchange_rate_time_series, 4),
                                G_previous2 = lag(exchange_rate_time_series, 3),
                                G_previous1 = lag(exchange_rate_time_series, 2),
                                G_current = lag(exchange_rate_time_series, 1),
                                G_pred = exchange_rate_time_series)

colnames(t6_time_lagged_data) <- c("Input1","Input2","Input3","Input4","Input5","Input6","Output")

# Remove the rows with NA
t6_time_lagged_data <- t6_time_lagged_data[complete.cases(t6_time_lagged_data), ]

View(t6_time_lagged_data)

#Scale the I/O Matrix (min-max normalization)
t6_normalise_data <- as.data.frame(lapply(t6_time_lagged_data,normalize))
head(t6_normalise_data)
summary(t6_normalise_data)

#Store the last index of the timed_lagged_data
t6_time_lagged_last_index <- nrow(t6_normalise_data)
t6_time_lagged_last_index

#Split the training and testing dataset
t6_train_data <- t6_normalise_data[1:400,]
t6_test_data <- t6_normalise_data[401:t6_time_lagged_last_index,]

#######################################################################################################################################################################

# MLP Model 1

t6_model <- neuralnet(Output ~ Input1 + Input2 + Input3 + Input4 + Input5 + Input6, 
                      data = t6_train_data, hidden = c(12,6), act.fct = "tanh",err.fct = 'sse',
                      linear.output = F)

plot(t6_model)

# Evaluating the model
# Generate predictions on the test data
t6_model_results <- compute(t6_model,t6_test_data[1:6])

# Obtain predicted result
t6_predicted_results <- t6_model_results$net.result
t6_predicted_results

t6_time_lagged_train <- t6_time_lagged_data[1:400,"Output"]
summary(t6_time_lagged_train)

t6_time_lagged_test <- t6_time_lagged_data[401:t6_time_lagged_last_index,"Output"]
summary(t6_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t6_min <- min(t6_time_lagged_train)
train_t6_max <- max(t6_time_lagged_train)

train_t6_max
train_t6_min

# de-normalize the normalized NN's output
t6_model_pred <- unnormalize(t6_predicted_results,train_t6_min,train_t6_max)
head(t6_model_pred)

#Calculate the Root Mean Squared Error(RMSE)
t6_rmse <- rmse(t6_time_lagged_test,t6_model_pred)
t6_rmse

# Mean Absolute Error (MAE)
t6_mae <- mae(t6_time_lagged_test,t6_model_pred)
t6_mae

# Mean Absolute Percentage Error (MAPE)
t6_mape <- mape(t6_time_lagged_test,t6_model_pred)
t6_mape

# Symmetric Mean Absolute Percentage Error (sMAPE)
t6_smape <- smape(t6_time_lagged_test,t6_model_pred)
t6_smape

t6_clean_output <- cbind(t6_time_lagged_test,t6_model_pred)
colnames(t6_clean_output) <- c("Output","Result")
t6_clean_output

# Visual Plot
par(mfrow=c(1,1))
plot(t6_time_lagged_test,t6_model_pred ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t6_time_lagged_test)
plot(x, t6_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t6_model_pred, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

###################################################################################################

# I/O Matrix (T-7)

####################################################################################################

t7_time_lagged_data <- bind_cols(G_previous6 = lag(exchange_rate_time_series, 7),
                                G_previous5 = lag(exchange_rate_time_series, 6),
                                G_previous4 = lag(exchange_rate_time_series, 5),
                                G_previous3 = lag(exchange_rate_time_series, 4),
                                G_previous2 = lag(exchange_rate_time_series, 3),
                                G_previous1 = lag(exchange_rate_time_series, 2),
                                G_current = lag(exchange_rate_time_series, 1),
                                G_pred = exchange_rate_time_series)

colnames(t7_time_lagged_data) <- c("Input1","Input2","Input3","Input4","Input5","Input6","Input7","Output")

# Remove the rows with NA
t7_time_lagged_data <- t7_time_lagged_data[complete.cases(t7_time_lagged_data), ]

View(t7_time_lagged_data)

#Store the last index of the timed_lagged_data
t7_time_lagged_last_index <- nrow(t7_time_lagged_data)
t7_time_lagged_last_index

#Scale the I/O Matrix (min-max normalization)
t7_normalise_data <- as.data.frame(lapply(t7_time_lagged_data,normalize))
head(t7_normalise_data)
summary(t7_normalise_data)


#Split the training and testing dataset
t7_train_data <- t7_normalise_data[1:400,]
t7_test_data <- t7_normalise_data[401:t7_time_lagged_last_index,]

#######################################################################################################################

# MLP Model 1

t7_model <- neuralnet(Output ~ Input1 + Input2 + Input3 + Input4 + Input5 + Input6 + Input7,
                      data = t7_train_data, hidden = 5, act.fct = "logistic",err.fct = 'sse',
                      linear.output = T)

plot(t7_model)

# Evaluating the model
# Generate predictions on the test data
t7_model_results <- compute(t7_model,t7_test_data[1:7])

# Obtain predicted result
t7_predicted_results <- t7_model_results$net.result
t7_predicted_results

t7_time_lagged_train <- t7_time_lagged_data[1:400,"Output"]
summary(t7_time_lagged_train)

t7_time_lagged_test <- t7_time_lagged_data[401:t7_time_lagged_last_index,"Output"]
summary(t7_time_lagged_test)

# Finding the maximum and minimum original training dataset
train_t7_min <- min(t7_time_lagged_train)
train_t7_max <- max(t7_time_lagged_train)

train_t7_max
train_t7_min

# de-normalize the normalized NN's output
t7_model_pred <- unnormalize(t7_predicted_results,train_t7_min,train_t7_max)
head(t7_model_pred)

#Calculate the Root Mean Squared Error(RMSE)
t7_rmse <- rmse(t7_time_lagged_test,t7_model_pred)
t7_rmse

# Mean Absolute Error (MAE)
t7_mae <- mae(t7_time_lagged_test,t7_model_pred)
t7_mae

# Mean Absolute Percentage Error (MAPE)
t7_mape <- mape(t7_time_lagged_test,t7_model_pred)
t7_mape

# Symmetric Mean Absolute Percentage Error (sMAPE)
t7_smape <- smape(t7_time_lagged_test,t7_model_pred)
t7_smape

t7_clean_output <- cbind(t7_time_lagged_test,t7_model_pred)
colnames(t7_clean_output) <- c("Output","Result")
t7_clean_output

# Visual Plot
par(mfrow=c(1,1))
plot(t7_time_lagged_test,t7_model_pred ,col='red',main='Real vs Predicted Exchange Rate',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)

x = 1:length(t7_time_lagged_test)
plot(x, t7_time_lagged_test, col = "red", type = "l", lwd=2,
     main = "Exchange Rate Prediction")
lines(x, t7_model_pred, col = "blue", lwd=2)
legend("bottomright",  legend = c("original exchange rate", "predicted exchange rate"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.1), cex = 0.6)
grid() 

#############################################################################################################################################

# Create a data frame with the model names and their respective values
model_data <- data.frame(
  "Model.Name" = c("t1_model_1", "t1_model_2", "t1_model_3", "t2_model_1", "t2_model_2", "t2_model_3", "t3_model_1", "t3_model_2", "t4_model_1","t4_model_2", "t5_model_1","t5_model_2", "t6_model", "t7_model"),
  "RMSE" = c(t1_rmse_1, t1_rmse_2, t1_rmse_3, t2_rmse_1, t2_rmse_2, t2_rmse_3, t3_rmse_1, t3_rmse_2, t4_rmse_1, t4_rmse_2, t5_rmse_1, t5_rmse_2, t6_rmse, t7_rmse),
  "MAE" = c(t1_mae_1, t1_mae_2, t1_mae_3, t2_mae_1, t2_mae_2, t2_mae_3, t3_mae_1, t3_mae_2, t4_mae_1, t4_mae_2, t5_mae_1, t5_mae_2, t6_mae, t7_mae),
  "MAPE" = c(t1_mape_1, t1_mape_2, t1_mape_3, t2_mape_1, t2_mape_2, t2_mape_3, t3_mape_1, t3_mape_2, t4_mape_1, t4_mape_2, t5_mape_1, t5_mape_2, t6_mape, t7_mape),
  "sMAPE" = c(t1_smape_1, t1_smape_2, t1_smape_3, t2_smape_1, t2_smape_2, t2_smape_3, t3_smape_1, t3_smape_2, t4_smape_1, t4_smape_2, t5_smape_1, t5_smape_2, t6_smape, t7_smape)
)

# Set the row names to the model names
row.names(model_data) <- model_data$Model

# Remove the redundant 'Model' column since we've set it as row names
model_data <- model_data[, -1]

# Print the data table
print(model_data)

