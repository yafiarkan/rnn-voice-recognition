# RNN Voice Recognition

Voice command recognition have been widely used for various needs such as helping people with disabilities, moving robots, and many other needs in daily day life. Many researchers have not done 100% approval, even with special tools such as Digital Signal Processing. The biggest challenge in the system is that everyone has a different tone and speech that will influence the predicted accuracy.

One of voice command recognition classifier is Reccurent Neural Network (RNN). RNN has several types, one of them is Long-Term Memory (LSTM). LSTM can recognize form sound signals which are sequential data or one-dimensional data. LSTM has a layer of memory to receive predictions of data learned, the forget gate to retrieve information that is not used so that it can make predictions more accurate in the future.

In this study, several types of testing are done in testing single words and words in sentences. This study resulted in an average evaluation 97% for this two types of test.

### Class
* Class 1 (Nyala)
* Class 2 (Mati)
* Class 3 (Lainnya)

### System Flow
The training system preview shown in the flow diagram below

![Training Flow](https://github.com/annisanazi/rnn-voice-recognition/blob/main/illustration%20flow.png)

The system illustration shown in the flow diagram below 

![System Flow](https://github.com/annisanazi/rnn-voice-recognition/blob/main/system_rnn.png)

The challenging part of this system is the system can predict the part or not. Most part sometimes contain only a position of the required data, for the example it should be "nyala" but the part only contain "nya" and "la" in the next part. 
