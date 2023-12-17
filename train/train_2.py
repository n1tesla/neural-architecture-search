import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras import backend as K
import csv
from controller import Controller, StateSpace
from manager import NetworkManager
from models.lstm_nas import model_fn


def train(config,X_train,y_train,X_test,y_test):

    # create a shared session between Keras and Tensorflow
    policy_sess = tf.compat.v1.Session()
    K.set_session(policy_sess)

    # construct a state space
    state_space = StateSpace()

    state_space.add_state(name='units',values=[64,128,256])
    state_space.print_state_space()

    dataset = [X_train, y_train, X_test, y_test]  # pack the dataset for the NetworkManager

    previous_acc = 0.0
    total_reward = 0.0

    with policy_sess.as_default():
        # create the Controller and build the internal policy network
        controller = Controller(policy_sess, config.NUM_LAYERS, state_space,
                                reg_param=config.REGULARIZATION,
                                exploration=config.EXPLORATION,
                                controller_cells=config.CONTROLLER_CELLS,
                                embedding_dim=config.EMBEDDING_DIM,
                                restore_controller=config.RESTORE_CONTROLLER)

    # create the Network Manager
    manager = NetworkManager(dataset, epochs=config.MAX_EPOCHS, child_batchsize=config.CHILD_BATCHSIZE, clip_rewards=config.CLIP_REWARDS,
                             acc_beta=config.ACCURACY_BETA)

    state = state_space.get_random_state_space(config.NUM_LAYERS)
    print("Initial Random State : ", state_space.parse_state_space_list(state))
    print()

    # clear the previous files
    controller.remove_files()

    # train for number of trails
    for trial in range(config.MAX_TRIALS):
        with policy_sess.as_default():
            K.set_session(policy_sess)
            actions = controller.get_action(state)  # get an action for the previous state

        # print the action probabilities
        state_space.print_actions(actions)
        print("Predicted actions : ", state_space.parse_state_space_list(actions))

        # build a models, train and get reward and accuracy from the network manager
        reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
        print("Rewards : ", reward, "Accuracy : ", previous_acc)

        with policy_sess.as_default():
            K.set_session(policy_sess)

            total_reward += reward
            print("Total reward : ", total_reward)

            # actions and states are equivalent, save the state and reward
            state = actions
            controller.store_rollout(state, reward)

            # train the controller on the saved state and the discounted rewards
            loss = controller.train_step()
            print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

            # write the results of this trial into a file
            with open('train_history.csv', mode='a+') as f:
                data = [previous_acc, reward]
                data.extend(state_space.parse_state_space_list(state))
                writer = csv.writer(f)
                writer.writerow(data)
        print()

    print("Total Reward : ", total_reward)