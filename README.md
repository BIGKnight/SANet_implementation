# SANET_IMPLEMENTATION
## implemented by Yezhen Wang through tensorflow
### PROLOGUE
Plainspoken, the current above implementation can't reach the results gotten by the author.And on the ShanghaiTech_B dataset, my SANet model only get 21.838 in MAE and 39.387 in MSE which far less than the 8.4 in MAE and 13.6 in MSE which mentioned in the papers.<br>Besides, I and my colleague both found a phenomena that once the network was added the instance normalization layer after each convolutional layer, then the model would output an uniform predicting density map readily.Thus, we both decided cut all the In layer dawn.  
### PREDICTION ON SHANGHAITECH_B DATASET
just put some good samples here<br>
![avatar](./result_image/2018-12-16%2017-18-52.png)
![avatar](./result_image/2018-12-16%2017-19-15.png)
![avatar](./result_image/2018-12-16%2017-20-01.png)
![avatar](./result_image/2018-12-16%2017-20-25.png)
![avatar](./result_image/2018-12-16%2017-21-25.png)
### GRAPH_STRUCTURE
the graph was obtained by using Tensorboard.So there are lots of tiny tensorflow op which you may not want to see.
<br> we can juse ignore those inessential ops and focus on the main backbone of the network.
![avatar](./result_image/graph.png)
### MAE AND RMSE TENDENCY CHART
The tendency chart of the validate set's MAE and MSE on the training model along with the changes of the time and training step
<br> Get by using Tensorboard
#### MAE tendency along with the training time
![avatar](./result_image/2018-12-16%2015-05-06.png)
#### MAE tendency along with tht step
![avatar](./result_image/2018-12-16%2015-05-50.png)
#### RMSE tendency along with the training time
![avatar](./result_image/2018-12-16%2015-07-01.png)
#### RMSE tendency along with the step
![avatar](./result_image/2018-12-16%2015-06-35.png)