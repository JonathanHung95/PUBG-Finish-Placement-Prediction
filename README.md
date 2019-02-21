# PUBG-Finish-Placement-Prediction

## Introduction

The goal of this repository is to contain all the various R files used for Kaggle's "PUBG Finish Placement Prediction" challenge.  The challenge's aim is to predict the win placement percentile of players, given data from the game, where 1 is a win and 0 is last place and everything in between is essentially the percentile placement of the player.

## The Datasets
The datasets can be obtained from here: https://www.kaggle.com/c/pubg-finish-placement-prediction/data .  It contains 6.38 million rows of data split into 4.45 million rows of training data and 1.93 million rows of testing data.  

## PUBG EDA
This is an md file written in Rstudio that conducts an exploratory data analysis of the dataset to investigate its various features.  

## PUBG Modelling Round 2
MD file written in Rstudio containing pre-processing done to the dataset, feature engineering and the model created to fit the training data.  Model and resulting data were uploaded to Kaggle for a Mean Average Error (MAE) of 0.0384.  Best position achieved was the 37th percentile.  Final percentile is 42nd.

## Learning Points
1. If we're going to have to process both the train and test data, it would be infinitely better to write the code to a function so that we don't have to rewrite the all the code to apply to both datasets.
