# LODsmartAR
Our goal in this project is to optimize the quality of experience in a AR/MR environment. The user experience is affected mainly by the quality of the rendered model perceived by the user and the update frequency of the 3D model as the user moves.

In order to optimize the QoE we need to trade off quality with update frequency. In order to achieve this, we use different Level of Details and we choose the best LOD that optimizes the user experience.

## SSIM predictor
The first step in our project is to evaluate the quality of the rendered model with metrics that best represent the quality as perceived by the user. In this first experiment I used the SSIM metric to represent the quality of a rendered 3D model from a specific camera position and orientation. The SSIM of a specific LOD is calculated by comparing it with the best version of the 3D model.\
In this first test I developed a deep learning model which takes in input the 3D coordinate of the camera position (we may include the camera orientation in the future, for now we assume it's looking at (0,0,0)) and the NN outputs the predicted SSIM for each LOD. Therefore the NN has 3 inputs and one output for each LOD.

### Dataset
In `random_walk_uniform.py` I generated 100 random camera positions around the center of the 3d model in a uniformly manner. This way we have a uniform sampling of the "complexity" of the 3d model viewed from any direction. These positions are then fed to Unity which takes a screenshot in every generated camera position in every LOD.
For example in the "LacockAbbey" 3d model we have 4 version of the mesh, we use the best version as reference and given a specific camera position we compare the reference LOD with the other LODs of the mesh resulting in 3 SSIMs.
As we can see in the plot below, the SSIMs tend to be higher as the camera is positioned farther from the center.

![ssim at different positions](documentation/images/2d_ssim_plot.png)

The correlation is more visible in the following plot which shows the distance of the camera from the center with respect to its ssim. We can also see that lower LOD have higher SSIMs as excpected (in our numeration LOD 0 is the closest to the Camera, and therefore the most detailed LOD level)
![distance with respect to ssim](documentation/images/ssim_vs_distance.png)

### NN architecture
In this test I used the following neural network architecture with the input layer consting of 3 nodes, 2 hidden layers consisting of 128 nodes each and an output layer with one node for each lod level.
```
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, len(LOD_NAMES)),
    nn.ReLU()
```

I split the dataset into 80 samples for training set and 20 samples for validation set, k-fold cross validation could be used in the future. For the loss function I chose the mean squared error which is I think is the most reasonable since we are working in eucledean space.\
In order to reduce overfitting I applied some regularization techniques such as dropout, l2 weight decay and reduce learning rate on plateau scheduler. 

### Results
After training for 200 epochs these are the results. The training took a very short time (in the order of few seconds). As we can see both training and validation loss appear to be very low which indicates a good fit of the model (not overfitting nor underfitting).
![training history](documentation/images/training_history.png)

In the following plot we try to infer the model by changing the x and z coordinate and keeping the y coordinate to one and we chose an arbitrary lod to plot the ssim at different positions. As we can see the model correctly learned that we have lower SSIMs (corresponding to lower quality) the closer the camera is positioned to the mesh.
![inference plot](documentation/images/ssim_inference_plot.png)

Here I report the prediction of a batch of validation samples outputted by the model:
```
example prediction: tensor([[1.0038, 0.9592, 0.8812],
        [0.9182, 0.8352, 0.7114],
        [0.9479, 0.8664, 0.7588],
        [0.9455, 0.8838, 0.7823]])
example label: tensor([[0.9573, 0.9149, 0.8520],
        [0.9260, 0.8424, 0.7204],
        [0.9430, 0.8681, 0.7632],
        [0.9509, 0.8849, 0.7774]])
```

## Positions predictor
During the rendering of the 3d model in a MR environment, the user is able to freely move around the 3d objects. In order to achieve our goal of optimizing the user experience we have to choose the optimal LOD to render which would give the perfect balance between visual quality and rendering speed. \
Let's assume that the user is positioned somewhere in space and is moving around, we have to optimize the visualization by picking the best LOD to render in the following frames. Therefore, we have to predict the movements of the user in order to know if the user is moving into a position from which the mesh is more or less complex.\
We have seen that by using the camera position we can accurately estimate the perceived quality of the rendered 3D model for each LOD. Therefore, if we can predict the next positions of the user we will be able to pick the best LOD to render in the successive frames.

### Dataset
In this first test I generated a simulated random walk consisting of 10k positions samples (see `random_walk.py`). The following image shows the plot of the random walk.
![random walk positions](documentation/images/random_walk.png)

###  NN architecture
Predicting the next camera positions is a time series analysis problem. The approach I took is by using a LSTM, taking in input a sequence of 20 latest camera positions and outputting the next camera position (we may want to predict multiple successive positions in future work).\
In this test I used 128 hidden features and two LSTM layers in series. ~~Regularizations are also applied here with dropout and l2 weight decay.~~ After a quick test with and without regularization, I found that without using dropout and weight decay is performing a bit better.
```
LSTM(
  (lstm): LSTM(3, 128, num_layers=2, batch_first=True)
  (linear): Linear(in_features=128, out_features=3, bias=True)
)
```

### Results
After training with Adam optimizer with learning rate starting from 1e-3 and MSE loss function, here is the training curve after 100 epochs.

![position predictor training history](documentation/images/pos_predictor_train_history.png)

At epoch 100 the training and validation loss are the following:
```
Train avg loss: 0.007497
Eval avg loss: 0.014971
```

Here I report the prediction of a batch of validation samples outputted by the model:
```
example prediction: tensor([[-5.2743e-03,  8.1226e-05,  5.0937e-01],
        [ 5.6479e-01,  7.1842e-05,  6.9548e-01],
        [ 6.0663e-01,  8.3087e-05,  6.9826e-01],
        [ 4.9452e-01, -1.5524e-04, -1.1854e-01]])
example label: tensor([[-0.0081,  0.0000,  0.5205],
        [ 0.5629,  0.0000,  0.7017],
        [ 0.6045,  0.0000,  0.7029],
        [ 0.5000,  0.0000, -0.1397]])
```

### Update 27/4/2021
For comparison, I implemented a naïve model which simply returns the last position of the input sequence, without predicting anything.
```
naive model train loss: 0.00713
naive model val loss: 0.0141
```
As we can see, both training and validation loss of our model is very close to the losses of the naïve model. This implies that our model isn't really predicting much. This could indicate an issue with our deep learning model, or that the random generated path is too random. Either way, further investigation is needed.

## Fps - vertex count correlation
During the rendering of 3d graphics it is intuitive that the larger the number of vertices to be rendered, the longer it takes to render a single frame, therefore the lower the refresh rate is. In the following graph we can clearly see this correlation between the fps and the log2 of the vertex count (as returned by `UnityEditor.UnityStats.vertices`) 
![dynamic ssim predictor training history](documentation/images/fps_vertex_count_correlation.png)
Below a certain vertex count the fps is limited to a range around 60 fps because the refresh rate of my display is 60Hz so the fps is capped to that frequency.\
Given this result, instead of predicting the fps which is unstable and depends on the performance of the device, we can predict instead the number of vertices to be rendered which is indipendent from the rendering device.

## Dynamic SSIM and vertex count predictor with generic 3d model
The quality of experience in a AR/MR environment is affected by the lag of the animation and the quality of the 3d model. The lag depends on two main factors: the refresh rate of the rendering and the speed of the camera as the user moves around the rendered mesh (when the camera is still, a low refresh rate in not noticeable). However it's incorrect to directly use the camera speed since a camera movement farther away from the 3d model matters less than a closer one with the same speed. \
In our approach we project the camera position a constant delta time into the future e.g. t=0.1s, and we calculate the difference in visual appearance between the frame in the current position and the one projected into the future, using the **SSIM index**. The faster the user is moving, the farther apart are the camera positions, therefore the lower the SSIM is. In order to evaluate the visual quality of the target LOD we take the frame in the current position using the highest LOD and the frame in the projected position using the target LOD.\
Considering the correlation between fps and vertex count in the previous result, in our model we try to predict the **vertex count** alongside the ssim score.

### NN architecture
![diagram](documentation/images/tesi_diagram.png)
```
MultimodelNN(
  (cnn): ConvNN(
    (cnn): Sequential(
      (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
      (8): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
      (9): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (10): ReLU()
      (11): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
      (12): Flatten(start_dim=1, end_dim=-1)
    )
  )
  (ffn): FeedForwardNN(
    (linear_relu_stack): Sequential(
      (0): Linear(in_features=15, out_features=256, bias=True)
      (1): ReLU()
      (2): Dropout(p=0, inplace=False)
      (3): Linear(in_features=256, out_features=256, bias=True)
      (4): ReLU()
      (5): Dropout(p=0, inplace=False)
      (6): Linear(in_features=256, out_features=2, bias=True)
    )
  )
)
```

```
Epoch 150
-------------------------------
Train avg loss: 0.000028
Eval avg loss: 0.000042 

example prediction: tensor([[0.5407, 0.6635],
        [0.9057, 0.6599],
        [0.9589, 0.6529],
        [1.0103, 0.8558]], device='cuda:0')
example label: tensor([[0.5305, 0.6688],
        [0.8985, 0.6561],
        [0.9646, 0.6527],
        [1.0000, 0.8549]], device='cuda:0')
```

The following test is executed on a 3d model which is not present in the training set. We can see that our network is able to generalize to different 3d meshes.
```
Test avg loss: 0.000569 

example prediction: tensor([[0.8715, 0.6653],
        [0.9630, 0.6510],
        [0.9876, 0.6161],
        [0.9816, 0.6163]], device='cuda:0')
example label: tensor([[0.7837, 0.6649],
        [0.9422, 0.6562],
        [0.9395, 0.5198],
        [0.9508, 0.5198]], device='cuda:0')
```

![dynamic ssim predictor training history](documentation/images/dyn_ssim_multimodel_training_history.png)