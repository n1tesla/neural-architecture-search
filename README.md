# Neural Architecture Search with Controller RNN

Basic implementation of Controller RNN from [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) and [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012).

- Uses Keras to define and train children / generated networks, which are defined in Tensorflow by the Controller RNN.
- Define a state space by using `StateSpace`, a manager which adds states and handles communication between the Controller RNN and the user.
- `Controller` manages the training and evaluation of the Controller RNN
- `NetworkManager` handles the training and reward computation of a Keras model

# Usage
At a high level : For full training details, please see `train.py`.

<img src="https://github.com/n1tesla/DeepAutoTraining/blob/main/images/start_training.png?raw=true" height=100% width=100%>

- Choose how deep are you going to search for hyperparameters: `Quick`, `Mid`, `Deep`
- Choose which problem are you going to solve it: `ANOMALY`, `REGRESSION`, `FORECASTING`, `CLASSIFICATION`
- Choose which models are you going to use for the problem: `LSTM_NAS`, `FCN`
- Type what you want to observe during training. Ex: Different features, window_size, scaler, hyperparameter etc. This observation name is going to be added to end of the network name and the folder will be created as `NetworkName_ObservationName` under `saved_models`.
- After the fine-tuning(training) finished, the best models are going to save with this pattern: `WindowSize_StrideSize_ScalerType_ValRatio`


```python
# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='kernel', values=[1, 3])
state_space.add_state(name='filters', values=[16, 32, 64])

# create the managers
controller = Controller(tf_session, num_layers, state_space)
manager = NetworkManager(dataset, epochs=max_epochs, batchsize=batchsize)

# For number of trials
sample_state = ...
actions = controller.get_actions(sample_state)
reward = manager.get_reward(actions)
controller.train_nas_()
```

# Implementation details
This is a very limited project with this version but it will develop very soon. 


# Problem with Evaluation Part
After training is completed I am getting a error in the below:

    raise TypeError(f'Argument `fetch` = {fetch} has invalid type '
`TypeError: Argument fetch = None has invalid type "NoneType". Cannot be None`

**Note**: I will try to fix the error as soon as possible. 

# Requirements
- Keras >= 1.2.1
- Tensorflow-gpu >= 1.2
