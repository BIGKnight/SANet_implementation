# SANet_implementation
## implemented by Yezhen Wang through tensorflow
### PROLOGUE
Plainspokenly, the current above implementation can't reach the results gotten by the author.And on the ShanghaiTech_B dataset, my SANet model only get 21.838 in MAE and 39.387 in MSE which far less than the 8.4 in MAE and 13.6 in MSE which mentioned in the papers.<br>Besides, I and my colleague both found a phenomena that once the network was added the instance normalization layer after each convolutional layer, then the model would output an uniform predicting density map readily.Thus, we both decided cut all the In layer dawn.  
