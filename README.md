# Predicting_F1_Pit_Stops
This repository contains my solution for the Kaggle Predicting F1 Pit Stops competition

## Overview
In Formula 1, pit stops are far more than just a mandatory tire change - they are the defining strategic element of a Grand Prix. A brilliantly timed pit stop can completely upend the race order, allowing a driver to outsmart faster cars and secure an unexpected victory. Accurately predicting when a driver will pit requires understanding a complex web of race dynamics, tire physics, and split-second team strategy.

## Goal
The goal is binary classification whether a Formula 1 driver will pit on the next lap.

## Dataset
Dataset for this competition was generated synthetically inspired by the real-world data (F1 strategy data). Feature distributions are close to the original ones.

- Dataset was split by organizers to train and test sets. 
- Target variable: `PitNextLap`

## Key Challenges and Strategic Approach
Before starting coding I observed 2 potential challenges:
- **Track-Dependent Lap Times:** Raw lap times (LapTime (s)) are highly circuit-specific and cannot be directly compared across the dataset. For example, an 80-second lap might indicate a slow pace at Monaco but is physically impossible at the Red Bull Ring. To prevent the model from misinterpreting raw pace, this solution focuses on relative feature engineering.
- **Chronological Data Leakage:** Formula 1 race data is inherently sequential. If the dataset undergoes a random train/test split, it introduces a severe risk of data leakage. For instance, the model could theoretically use information from lap 19 in the training set to predict lap 18 of the same race in the test set. To build a robust model that learns genuine pit strategies rather than memorizing specific historical events, the cross-validation strategy relies on grouped splitting. 
**Edit:** An exploratory analysis of the dataset revealed a complete overlap in the `Race + Year` feature between the training and test sets (104 out of 104 races are shared). From a real-world Formula 1 forecasting perspective, this is a highly unrealistic split. In practice, predicting pit stops requires forecasting future events based entirely on past races. 


