# TrumpTweetGenerator
A simple recurrent neural network generating Trump Tweets

## Installation

- Install python3
- Install tensorflow 
```
  pip3 install tensorflow
 ```
 
 ## Usage
 
 ### 1. Train Model:
 
**! This repositoy comes with a pre-trained Neural Network so you dont have to train it yourself for some quick fun !**
 
   adjust line 59 of main.py to a list of Tweets from Trump (or whoever you want) separated with an "ß".
   Call main.py with the t parameter:
```
  python3 main.py -t
```


### 2. Generate Tweets:
  
  Create a File named Tweets.txt. Per default all tweets will be saved there.
  Just call the main.py without the t parameter. 
  ```
  python3 main.py
  ```
  For further explanation of the available parameter just use the h parameter:
  ```
  python3 main.py -h
  ```
