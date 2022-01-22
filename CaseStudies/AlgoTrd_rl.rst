.. _AlgoTrd_rl:


Reinforcement Learning based Trading Strategy
=============================================

In this case study, we will create an end-to-end trading strategy based
on Reinforcement Learning.

Content
-------

-  `1. Problem Definition <#0>`__
-  `2. Getting Started - Load Libraries and Dataset <#1>`__

   -  `2.1. Load Libraries <#1.1>`__
   -  `2.2. Load Dataset <#1.2>`__

-  `3. Exploratory Data Analysis <#2>`__

   -  `3.1 Descriptive Statistics <#2.1>`__
   -  `3.2. Data Visualisation <#2.2>`__

-  `4. Data Preparation <#3>`__

   -  `4.1 Data Cleaning <#3.1>`__

-  `5.Evaluate Algorithms and Models <#5>`__

   -  `5.1. Train Test Split <#5.1>`__
   -  `5.2. Implementation steps and modules <#5.2>`__
   -  `5.3. Agent Script <#5.3>`__
   -  `5.4. Helper Function <#5.4>`__
   -  `5.5. Training Set <#5.5>`__

-  `6.Test Set <#6>`__

 # 1. Problem Definition

In this Reinforcement Learning framework for trading strategy, the
algorithm takes an action (buy, sell or hold) depending upon the current
state of the stock price. The algorithm is trained using Deep Q-Learning
framework, to help us predict the best action, based on the current
stock prices.

The key components of the RL based framework are : \* Agent: Trading
agent. \* Action: Buy, sell or hold. \* Reward function: Realized profit
and loss (PnL) is used as the reward function for this case study. The
reward depends upon the action: \* Sell: Realized profit and loss (sell
price - bought price) \* Buy: No reward \* Hold: No Reward

-  State: Differences of past stock prices for a given time window is
   used as the state.

The data used for this case study will be the standard and poor’s 500.
The link to the data is :
https://ca.finance.yahoo.com/quote/%255EGSPC/history?p=%255EGSPC).

 # 2. Getting Started- Loading the data and python packages

 ## 2.1. Loading the python packages

.. code:: ipython3

    # Load libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas import read_csv, set_option
    from pandas.plotting import scatter_matrix
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    import datetime
    import math
    from numpy.random import choice
    import random

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    #Import Model Packages for reinforcement learning
    from keras import layers, models, optimizers
    from keras import backend as K
    from collections import namedtuple, deque

 ## 2.2. Loading the Data

.. code:: ipython3

    #The data already obtained from yahoo finance is imported.
    dataset = read_csv('data/SP500.csv',index_col=0)

.. code:: ipython3

    #Diable the warnings
    import warnings
    warnings.filterwarnings('ignore')

.. code:: ipython3

    type(dataset)




.. parsed-literal::

    pandas.core.frame.DataFrame



 # 3. Exploratory Data Analysis

.. code:: ipython3

    # shape
    dataset.shape




.. parsed-literal::

    (2516, 6)



.. code:: ipython3

    # peek at data
    set_option('display.width', 100)
    dataset.head(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Adj Close</th>
          <th>Volume</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2010-01-04</th>
          <td>1116.56</td>
          <td>1133.87</td>
          <td>1116.56</td>
          <td>1132.99</td>
          <td>1132.99</td>
          <td>3991400000</td>
        </tr>
        <tr>
          <th>2010-01-05</th>
          <td>1132.66</td>
          <td>1136.63</td>
          <td>1129.66</td>
          <td>1136.52</td>
          <td>1136.52</td>
          <td>2491020000</td>
        </tr>
        <tr>
          <th>2010-01-06</th>
          <td>1135.71</td>
          <td>1139.19</td>
          <td>1133.95</td>
          <td>1137.14</td>
          <td>1137.14</td>
          <td>4972660000</td>
        </tr>
        <tr>
          <th>2010-01-07</th>
          <td>1136.27</td>
          <td>1142.46</td>
          <td>1131.32</td>
          <td>1141.69</td>
          <td>1141.69</td>
          <td>5270680000</td>
        </tr>
        <tr>
          <th>2010-01-08</th>
          <td>1140.52</td>
          <td>1145.39</td>
          <td>1136.22</td>
          <td>1144.98</td>
          <td>1144.98</td>
          <td>4389590000</td>
        </tr>
      </tbody>
    </table>
    </div>



The data has total 2515 rows and six columns which contain the open,
high, low, close and adjusted close price along with the total volume.
The adjusted close is the closing price adjusted for the split and
dividends. For the purpose of this case study, we will be focusing on
the closing price.

.. code:: ipython3

    # describe data
    set_option('precision', 3)
    dataset.describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Adj Close</th>
          <th>Volume</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>2516.000</td>
          <td>2516.000</td>
          <td>2516.000</td>
          <td>2516.000</td>
          <td>2516.000</td>
          <td>2.516e+03</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>1962.148</td>
          <td>1971.347</td>
          <td>1952.200</td>
          <td>1962.609</td>
          <td>1962.609</td>
          <td>3.715e+09</td>
        </tr>
        <tr>
          <th>std</th>
          <td>589.031</td>
          <td>590.191</td>
          <td>587.624</td>
          <td>588.910</td>
          <td>588.910</td>
          <td>8.134e+08</td>
        </tr>
        <tr>
          <th>min</th>
          <td>1027.650</td>
          <td>1032.950</td>
          <td>1010.910</td>
          <td>1022.580</td>
          <td>1022.580</td>
          <td>1.025e+09</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>1381.643</td>
          <td>1390.700</td>
          <td>1372.800</td>
          <td>1384.405</td>
          <td>1384.405</td>
          <td>3.238e+09</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>1985.320</td>
          <td>1993.085</td>
          <td>1975.660</td>
          <td>1986.480</td>
          <td>1986.480</td>
          <td>3.588e+09</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>2434.180</td>
          <td>2441.523</td>
          <td>2427.960</td>
          <td>2433.968</td>
          <td>2433.968</td>
          <td>4.077e+09</td>
        </tr>
        <tr>
          <th>max</th>
          <td>3247.230</td>
          <td>3247.930</td>
          <td>3234.370</td>
          <td>3240.020</td>
          <td>3240.020</td>
          <td>1.062e+10</td>
        </tr>
      </tbody>
    </table>
    </div>



Let us look at the plot of the stock movement.

.. code:: ipython3

    dataset['Close'].plot()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1f2b0064da0>




.. image:: output_18_1.png


 ## 4. Data Preparation

 ## 4.1. Data Cleaning Let us check for the NAs in the rows, either drop
them or fill them with the mean of the column

.. code:: ipython3

    #Checking for any null values and removing the null values'''
    print('Null Values =',dataset.isnull().values.any())


.. parsed-literal::

    Null Values = False


In case there are null values fill the missing values with the last
value available in the dataset.

.. code:: ipython3

    # Fill the missing values with the last value available in the dataset.
    dataset=dataset.fillna(method='ffill')
    dataset.head(2)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Adj Close</th>
          <th>Volume</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2010-01-04</th>
          <td>1116.56</td>
          <td>1133.87</td>
          <td>1116.56</td>
          <td>1132.99</td>
          <td>1132.99</td>
          <td>3991400000</td>
        </tr>
        <tr>
          <th>2010-01-05</th>
          <td>1132.66</td>
          <td>1136.63</td>
          <td>1129.66</td>
          <td>1136.52</td>
          <td>1136.52</td>
          <td>2491020000</td>
        </tr>
      </tbody>
    </table>
    </div>



The parameters to clusters are the indices and the variables used in the
clustering are the columns. Hence the data is in the right format to be
fed to the clustering algorithms

 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split

We will use 80% of the dataset for modeling and use 20% for testing.

.. code:: ipython3

    X=list(dataset["Close"])
    X=[float(x) for x in X]

.. code:: ipython3

    validation_size = 0.2
    #In case the data is not dependent on the time series, then train and test split should be done based on sequential sample
    #This can be done by selecting an arbitrary split point in the ordered list of observations and creating two new datasets.
    train_size = int(len(X) * (1-validation_size))
    X_train, X_test = X[0:train_size], X[train_size:len(X)]

 ## 5.2. Implementation steps and modules

The algorithm, in simple terms decides whether to buy, sell or hold,
when provided with the current market price. The algorithm is based on
“Q-learning based” approach and used Deep-Q-Network (DQN) to come up
with a policy. As discussed before, the name “Q-learning” comes from the
Q(s, a) function, that based on the state s and provided action a
returns the expected reward.

In order to implement this DQN algorithm several functions and modules
are implemented that interact with each other during the model training.
A summary of the modules and functions is described below.

1. **Agent Class**: The agent is defined as “Agent” class, that holds
   the variables and member functions that perform the Q-Learning that
   we discussed before. An object of the “Agent” class is created using
   the training phase and is used for training the model.
2. **Helper functions**: In this module, we create additional functions
   that are helpful for training. There are two helper functions that we
   have are as follows.
3. **Training module**: In this step, we perform the training of the
   data using the vari‐ ables and the functions agent and helper
   methods. This will provide us with one of three actions (i.e. buy,
   sell or hold) based on the states of the stock prices at the end of
   the day. During training, the prescribed action for each day is
   predicted, the rewards are computed and the deep-learning based
   Q-learning model weights are updated iteratively over a number of
   episodes. Additionally, the profit and loss of each action is summed
   up to see whether an overall profit has occur‐ red. The aim is to
   maximize the total profit. We provide a deep dive into the
   interaction between different modules and functions in the “Training
   the model” section below. Let us look at the each of the modules in
   detail

 ## 5.3. Agent script

The definition of the Agent script is the key step, as it consists of
the In this section, we will train an agent that will perform
reinforcement learning based on the Q-Learning. We will perform the
following steps to achieve this:

-  Create an agent class whose initial function takes in the batch size,
   state size, and an evaluation Boolean function, to check whether the
   training is ongoing.
-  In the agent class, create the following methods:

   -  Constructor: The constructor inititalises all the parameters.
   -  Model : This f unction has a deep learning model to map the state
      to action.
   -  Act function :Returns an action, given a state, using the output
      of the model function. The number of actions are defined as 3:
      sit, buy, sell
   -  expReplay : Create a Replay function that adds, samples, and
      evaluates a buffer. Add a new experience to the replay buffer
      memory. Randomly sample a batch of experienced tuples from the
      memory. In the following function, we randomly sample states from
      a memory buffer. Experience replay stores a history of state,
      action, reward, and next state transitions that are experienced by
      the agent. It randomly samples mini-batches from this experience
      to update the network weights at each time step before the agent
      selects an ε-greedy action.

Experience replay increases sample efficiency, reduces the
autocorrelation of samples that are collected during online learning,
and limits the feedback due to the current weights producing training
samples that can lead to local minima or divergence.

.. code:: ipython3

    import keras
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers import Dense
    from keras.optimizers import Adam
    from IPython.core.debugger import set_trace

    import numpy as np
    import random
    from collections import deque

    class Agent:
        def __init__(self, state_size, is_eval=False, model_name=""):
            #State size depends and is equal to the the window size, n previous days
            self.state_size = state_size # normalized previous days,
            self.action_size = 3 # sit, buy, sell
            self.memory = deque(maxlen=1000)
            self.inventory = []
            self.model_name = model_name
            self.is_eval = is_eval

            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            #self.epsilon_decay = 0.9

            #self.model = self._model()

            self.model = load_model(model_name) if is_eval else self._model()

        #Deep Q Learning model- returns the q-value when given state as input
        def _model(self):
            model = Sequential()
            #Input Layer
            model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
            #Hidden Layers
            model.add(Dense(units=32, activation="relu"))
            model.add(Dense(units=8, activation="relu"))
            #Output Layer
            model.add(Dense(self.action_size, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=0.001))
            return model

        #Return the action on the value function
        #With probability (1-$\epsilon$) choose the action which has the highest Q-value.
        #With probability ($\epsilon$) choose any action at random.
        #Intitially high epsilon-more random, later less
        #The trained agents were evaluated by different initial random condition
        #and an e-greedy policy with epsilon 0.05. This procedure is adopted to minimize the possibility of overfitting during evaluation.

        def act(self, state):
            #If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
            #actions suggested.
            if not self.is_eval and random.random() <= self.epsilon:
                return random.randrange(self.action_size)
            options = self.model.predict(state)
            #set_trace()
            #action is based on the action that has the highest value from the q-value function.
            return np.argmax(options[0])

        def expReplay(self, batch_size):
            mini_batch = []
            l = len(self.memory)
            for i in range(l - batch_size + 1, l):
                mini_batch.append(self.memory[i])

            # the memory during the training phase.
            for state, action, reward, next_state, done in mini_batch:
                target = reward # reward or Q at time t
                #update the Q table based on Q table equation
                #set_trace()
                if not done:
                    #set_trace()
                    #max of the array of the predicted.
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # Q-value of the state currently from the table
                target_f = self.model.predict(state)
                # Update the output Q table for the given action in the table
                target_f[0][action] = target
                #train and fit the model where state is X and target_f is Y, where the target is updated.
                self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

 ## 5.4. Helper Functions

In this script, we will create functions that will be helpful for
training. We create the following functions:

1) formatPrice:format the price to two decimal places, to reduce the
   ambiguity of the data:

2) getStockData: Return a vector of stock data from the CSV file.
   Convert the closing stock prices from the data to vectors, and return
   a vector of all stock prices.

3) getState: Define a function to generate states from the input vector.
   Create the time series by generating the states from the vectors
   created in the previous step. The function for this takes three
   parameters: the data; a time, t (the day that you want to predict);
   and a window (how many days to go back in time). The rate of change
   between these vectors will then be measured and based on the sigmoid
   function.

.. code:: ipython3

    import numpy as np
    import math

    # prints formatted price
    def formatPrice(n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

    # # returns the vector containing stock data from a fixed file
    # def getStockData(key):
    #     vec = []
    #     lines = open("data/" + key + ".csv", "r").read().splitlines()

    #     for line in lines[1:]:
    #         vec.append(float(line.split(",")[4])) #Only Close column

    #     return vec

    # returns the sigmoid
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # returns an an n-day state representation ending at time t

    def getState(data, t, n):
        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
        #block is which is the for [1283.27002, 1283.27002]
        res = []
        for i in range(n - 1):
            res.append(sigmoid(block[i + 1] - block[i]))
        return np.array([res])

    # Plots the behavior of the output
    def plot_behavior(data_input, states_buy, states_sell, profit):
        fig = plt.figure(figsize = (15,5))
        plt.plot(data_input, color='r', lw=2.)
        plt.plot(data_input, '^', markersize=10, color='m', label = 'Buying signal', markevery = states_buy)
        plt.plot(data_input, 'v', markersize=10, color='k', label = 'Selling signal', markevery = states_sell)
        plt.title('Total gains: %f'%(profit))
        plt.legend()
        #plt.savefig('output/'+name+'.png')
        plt.show()

 ## 5.5. Training the data

We will proceed to train the data, based on our agent and helper
methods. This will provide us with one of three actions, based on the
states of the stock prices at the end of the day. These states can be to
buy, sell, or hold. During training, the prescribed action for each day
is predicted, and the price (profit, loss, or unchanged) of the action
is calculated. The cumulative sum will be calculated at the end of the
training period, and we will see whether there has been a profit or a
loss. The aim is to maximize the total profit.

Steps: \* Define the number of market days to consider as the window
size and define the batch size with which the neural network will be
trained. \* Instantiate the stock agent with the window size and batch
size. \* Read the training data from the CSV file, using the helper
function. \* The episode count is defined. The agent will look at the
data for so many numbers of times. An episode represents a complete pass
over the data. \* We can start to iterate through the episodes. \* Each
episode has to be started with a state based on the data and window
size. The inventory of stocks is initialized before going through the
data. \* **Start to iterate over every day of the stock data. The action
probability is predicted by the agent**. \* Next, every day of trading
is iterated, and the agent can act upon the data. Every day, the agent
decides an action. Based on the action, the stock is held, sold, or
bought. \* If the action is 1, then agent buys the stock. \* If the
action is 2, the agent sells the stocks and removes it from the
inventory. Based on the sale, the profit (or loss) is calculated.

-  If the action is 0, then there is no trade. The state can be called
   holding during that period.
-  The details of the state, next state, action etc is saved in the
   memory of the agent object, which is used further by the exeReply
   function.

.. code:: ipython3

    from IPython.core.debugger import set_trace
    window_size = 1
    agent = Agent(window_size)
    #In this step we feed the closing value of the stock price
    data = X_train
    l = len(data) - 1
    #
    batch_size = 32
    #An episode represents a complete pass over the data.
    episode_count = 10

    for e in range(episode_count + 1):
        print("Running episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)
        #set_trace()
        total_profit = 0
        agent.inventory = []
        states_sell = []
        states_buy = []
        for t in range(l):
            action = agent.act(state)
            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1: # buy
                agent.inventory.append(data[t])
                states_buy.append(t)
                #print("Buy: " + formatPrice(data[t]))

            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                states_sell.append(t)
                #print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

            done = True if t == l - 1 else False
            #appends the details of the state action etc in the memory, which is used further by the exeReply function
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")
                #set_trace()
                #pd.DataFrame(np.array(agent.memory)).to_csv("Agent"+str(e)+".csv")
                #Chart to show how the model performs with the stock goin up and down for each
                plot_behavior(data,states_buy, states_sell, total_profit)
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)


        if e % 2 == 0:
            agent.model.save("model_ep" + str(e))


.. parsed-literal::

    Running episode 0/10
    --------------------------------
    Total Profit: $2179.84
    --------------------------------



.. image:: output_40_1.png


.. parsed-literal::

    Running episode 1/10
    --------------------------------
    Total Profit: -$45.07
    --------------------------------



.. image:: output_40_3.png


.. parsed-literal::

    Running episode 2/10
    --------------------------------
    Total Profit: $312.55
    --------------------------------



.. image:: output_40_5.png


.. parsed-literal::

    Running episode 3/10
    --------------------------------
    Total Profit: $13.25
    --------------------------------



.. image:: output_40_7.png


.. parsed-literal::

    Running episode 4/10
    --------------------------------
    Total Profit: $727.84
    --------------------------------



.. image:: output_40_9.png


.. parsed-literal::

    Running episode 5/10
    --------------------------------
    Total Profit: $535.26
    --------------------------------



.. image:: output_40_11.png


.. parsed-literal::

    Running episode 6/10
    --------------------------------
    Total Profit: $1290.32
    --------------------------------



.. image:: output_40_13.png


.. parsed-literal::

    Running episode 7/10
    --------------------------------
    Total Profit: $898.78
    --------------------------------



.. image:: output_40_15.png


.. parsed-literal::

    Running episode 8/10
    --------------------------------
    Total Profit: $353.15
    --------------------------------



.. image:: output_40_17.png


.. parsed-literal::

    Running episode 9/10
    --------------------------------
    Total Profit: $1971.54
    --------------------------------



.. image:: output_40_19.png


.. parsed-literal::

    Running episode 10/10
    --------------------------------
    Total Profit: $1926.84
    --------------------------------



.. image:: output_40_21.png


.. code:: ipython3

    #Deep Q-Learning Model
    print(agent.model.summary())


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    dense_13 (Dense)             (None, 64)                128
    _________________________________________________________________
    dense_14 (Dense)             (None, 32)                2080
    _________________________________________________________________
    dense_15 (Dense)             (None, 8)                 264
    _________________________________________________________________
    dense_16 (Dense)             (None, 3)                 27
    =================================================================
    Total params: 2,499
    Trainable params: 2,499
    Non-trainable params: 0
    _________________________________________________________________
    None


 # 6. Testing the Data

After training the data, it is tested it against the test dataset. Our
model resulted in a overall profit. The best thing about the model was
that the profits kept improving over time, indicating that it was
learning well and taking better actions.

.. code:: ipython3

    #agent is already defined in the training set above.
    test_data = X_test
    l_test = len(test_data) - 1
    state = getState(test_data, 0, window_size + 1)
    total_profit = 0
    is_eval = True
    done = False
    states_sell_test = []
    states_buy_test = []
    #Get the trained model
    model_name = "model_ep"+str(episode_count)
    agent = Agent(window_size, is_eval, model_name)
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

.. code:: ipython3

    for t in range(l_test):
        action = agent.act(state)
        #print(action)
        #set_trace()
        next_state = getState(test_data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(test_data[t])
            states_buy_test.append(t)
            print("Buy: " + formatPrice(test_data[t]))

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(test_data[t] - bought_price, 0)
            #reward = test_data[t] - bought_price
            total_profit += test_data[t] - bought_price
            states_sell_test.append(t)
            print("Sell: " + formatPrice(test_data[t]) + " | profit: " + formatPrice(test_data[t] - bought_price))

        if t == l_test - 1:
            done = True
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("------------------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("------------------------------------------")

    plot_behavior(test_data,states_buy_test, states_sell_test, total_profit)


.. parsed-literal::

    Buy: $2673.61
    Sell: $2695.81 | profit: $22.20
    Buy: $2748.23
    Sell: $2767.56 | profit: $19.33
    Buy: $2776.42
    Sell: $2802.56 | profit: $26.14
    Buy: $2798.03
    Sell: $2810.30 | profit: $12.27
    Buy: $2837.54
    Sell: $2839.25 | profit: $1.71
    Buy: $2853.53
    Buy: $2822.43
    Sell: $2823.81 | profit: -$29.72
    Buy: $2821.98
    Buy: $2762.13
    Buy: $2648.94
    Sell: $2695.14 | profit: -$127.29
    Buy: $2681.66
    Buy: $2581.00
    Sell: $2619.55 | profit: -$202.43
    Sell: $2656.00 | profit: -$106.13
    Sell: $2662.94 | profit: $14.00
    Sell: $2698.63 | profit: $16.97
    Sell: $2731.20 | profit: $150.20
    Buy: $2716.26
    Buy: $2701.33
    Sell: $2703.96 | profit: -$12.30
    Sell: $2747.30 | profit: $45.97
    Buy: $2744.28
    Buy: $2713.83
    Buy: $2677.67
    Sell: $2691.25 | profit: -$53.03
    Sell: $2720.94 | profit: $7.11
    Sell: $2728.12 | profit: $50.45
    Buy: $2726.80
    Sell: $2738.97 | profit: $12.17
    Buy: $2783.02
    Buy: $2765.31
    Buy: $2749.48
    Buy: $2747.33
    Sell: $2752.01 | profit: -$31.01
    Buy: $2712.92
    Sell: $2716.94 | profit: -$48.37
    Buy: $2711.93
    Buy: $2643.69
    Buy: $2588.26
    Sell: $2658.55 | profit: -$90.93
    Buy: $2612.62
    Buy: $2605.00
    Sell: $2640.87 | profit: -$106.46
    Buy: $2581.88
    Sell: $2614.45 | profit: -$98.47
    Sell: $2644.69 | profit: -$67.24
    Sell: $2662.84 | profit: $19.15
    Buy: $2604.47
    Sell: $2613.16 | profit: $24.90
    Sell: $2656.87 | profit: $44.25
    Buy: $2642.19
    Sell: $2663.99 | profit: $58.99
    Buy: $2656.30
    Sell: $2677.84 | profit: $95.96
    Sell: $2706.39 | profit: $101.92
    Sell: $2708.64 | profit: $66.45
    Buy: $2693.13
    Buy: $2670.14
    Buy: $2670.29
    Buy: $2634.56
    Sell: $2639.40 | profit: -$16.90
    Sell: $2666.94 | profit: -$26.19
    Sell: $2669.91 | profit: -$0.23
    Buy: $2648.05
    Sell: $2654.80 | profit: -$15.49
    Buy: $2635.67
    Buy: $2629.73
    Sell: $2663.42 | profit: $28.86
    Sell: $2672.63 | profit: $24.58
    Buy: $2671.92
    Sell: $2697.79 | profit: $62.12
    Sell: $2723.07 | profit: $93.34
    Sell: $2727.72 | profit: $55.80
    Buy: $2711.45
    Sell: $2722.46 | profit: $11.01
    Buy: $2720.13
    Buy: $2712.97
    Sell: $2733.01 | profit: $12.88
    Buy: $2724.44
    Sell: $2733.29 | profit: $20.32
    Buy: $2727.76
    Buy: $2721.33
    Buy: $2689.86
    Sell: $2724.01 | profit: -$0.43
    Buy: $2705.27
    Sell: $2734.62 | profit: $6.86
    Sell: $2746.87 | profit: $25.54
    Sell: $2748.80 | profit: $58.94
    Sell: $2772.35 | profit: $67.08
    Buy: $2770.37
    Sell: $2779.03 | profit: $8.66
    Buy: $2775.63
    Sell: $2782.49 | profit: $6.86
    Buy: $2779.66
    Buy: $2773.75
    Buy: $2762.59
    Sell: $2767.32 | profit: -$12.34
    Buy: $2749.76
    Sell: $2754.88 | profit: -$18.87
    Buy: $2717.07
    Sell: $2723.06 | profit: -$39.53
    Buy: $2699.63
    Sell: $2716.31 | profit: -$33.45
    Sell: $2718.37 | profit: $1.30
    Sell: $2726.71 | profit: $27.08
    Buy: $2713.22
    Sell: $2736.61 | profit: $23.39
    Buy: $2774.02
    Sell: $2798.29 | profit: $24.27
    Buy: $2798.43
    Sell: $2809.55 | profit: $11.12
    Buy: $2804.49
    Buy: $2801.83
    Sell: $2806.98 | profit: $2.49
    Sell: $2820.40 | profit: $18.57
    Buy: $2837.44
    Buy: $2818.82
    Buy: $2802.60
    Sell: $2816.29 | profit: -$21.15
    Buy: $2813.36
    Sell: $2827.22 | profit: $8.40
    Sell: $2840.35 | profit: $37.75
    Sell: $2850.40 | profit: $37.04
    Buy: $2857.70
    Buy: $2853.58
    Buy: $2833.28
    Buy: $2821.93
    Sell: $2839.96 | profit: -$17.74
    Buy: $2818.37
    Sell: $2840.69 | profit: -$12.89
    Sell: $2850.13 | profit: $16.85
    Sell: $2857.05 | profit: $35.12
    Sell: $2862.96 | profit: $44.59
    Buy: $2861.82
    Buy: $2856.98
    Sell: $2874.69 | profit: $12.87
    Sell: $2896.74 | profit: $39.76
    Buy: $2901.13
    Sell: $2901.52 | profit: $0.39
    Buy: $2896.72
    Buy: $2888.60
    Buy: $2878.05
    Buy: $2871.68
    Sell: $2877.13 | profit: -$19.59
    Sell: $2887.89 | profit: -$0.71
    Sell: $2888.92 | profit: $10.87
    Sell: $2904.18 | profit: $32.50
    Buy: $2888.80
    Sell: $2904.31 | profit: $15.51
    Buy: $2929.67
    Buy: $2919.37
    Buy: $2915.56
    Buy: $2905.97
    Sell: $2914.00 | profit: -$15.67
    Buy: $2913.98
    Sell: $2924.59 | profit: $5.22
    Buy: $2923.43
    Sell: $2925.51 | profit: $9.95
    Buy: $2901.61
    Buy: $2885.57
    Buy: $2884.43
    Buy: $2880.34
    Buy: $2785.68
    Buy: $2728.37
    Sell: $2767.13 | profit: -$138.84
    Buy: $2750.79
    Sell: $2809.92 | profit: -$104.06
    Buy: $2809.21
    Buy: $2768.78
    Buy: $2767.78
    Buy: $2755.88
    Buy: $2740.69
    Buy: $2656.10
    Sell: $2705.57 | profit: -$217.86
    Buy: $2658.69
    Buy: $2641.25
    Sell: $2682.63 | profit: -$218.98
    Sell: $2711.74 | profit: -$173.83
    Sell: $2740.37 | profit: -$144.06
    Buy: $2723.06
    Sell: $2738.31 | profit: -$142.03
    Sell: $2755.45 | profit: -$30.23
    Sell: $2813.89 | profit: $85.52
    Buy: $2806.83
    Buy: $2781.01
    Buy: $2726.22
    Buy: $2722.18
    Buy: $2701.58
    Sell: $2730.20 | profit: -$20.59
    Sell: $2736.27 | profit: -$72.94
    Buy: $2690.73
    Buy: $2641.89
    Sell: $2649.93 | profit: -$118.85
    Buy: $2632.56
    Sell: $2673.45 | profit: -$94.33
    Sell: $2682.17 | profit: -$73.71
    Sell: $2743.79 | profit: $3.10
    Buy: $2737.80
    Sell: $2760.17 | profit: $104.07
    Sell: $2790.37 | profit: $131.68
    Buy: $2700.06
    Buy: $2695.95
    Buy: $2633.08
    Sell: $2637.72 | profit: -$3.53
    Buy: $2636.78
    Sell: $2651.07 | profit: -$71.99
    Buy: $2650.54
    Buy: $2599.95
    Buy: $2545.94
    Sell: $2546.16 | profit: -$260.67
    Buy: $2506.96
    Buy: $2467.42
    Buy: $2416.62
    Buy: $2351.10
    Sell: $2467.70 | profit: -$313.31
    Sell: $2488.83 | profit: -$237.39
    Buy: $2485.74
    Sell: $2506.85 | profit: -$215.33
    Sell: $2510.03 | profit: -$191.55
    Buy: $2447.89
    Sell: $2531.94 | profit: -$158.79
    Sell: $2549.69 | profit: -$92.20
    Sell: $2574.41 | profit: -$58.15
    Sell: $2584.96 | profit: -$152.84
    Sell: $2596.64 | profit: -$103.42
    Buy: $2596.26
    Buy: $2582.61
    Sell: $2610.30 | profit: -$85.65
    Sell: $2616.10 | profit: -$16.98
    Sell: $2635.96 | profit: -$0.82
    Sell: $2670.71 | profit: $20.17
    Buy: $2632.90
    Sell: $2638.70 | profit: $38.75
    Sell: $2642.33 | profit: $96.39
    Sell: $2664.76 | profit: $157.80
    Buy: $2643.85
    Buy: $2640.00
    Sell: $2681.05 | profit: $213.63
    Sell: $2704.10 | profit: $287.48
    Sell: $2706.53 | profit: $355.43
    Sell: $2724.87 | profit: $239.13
    Sell: $2737.70 | profit: $289.81
    Buy: $2731.61
    Buy: $2706.05
    Sell: $2707.88 | profit: $111.62
    Sell: $2709.80 | profit: $127.19
    Sell: $2744.73 | profit: $111.83
    Sell: $2753.03 | profit: $109.18
    Buy: $2745.73
    Sell: $2775.60 | profit: $135.60
    Sell: $2779.76 | profit: $48.15
    Sell: $2784.70 | profit: $78.65
    Buy: $2774.88
    Sell: $2792.67 | profit: $46.94
    Sell: $2796.11 | profit: $21.23
    Buy: $2793.90
    Buy: $2792.38
    Buy: $2784.49
    Sell: $2803.69 | profit: $9.79
    Buy: $2792.81
    Buy: $2789.65
    Buy: $2771.45
    Buy: $2748.93
    Buy: $2743.07
    Sell: $2783.30 | profit: -$9.08
    Sell: $2791.52 | profit: $7.03
    Sell: $2810.92 | profit: $18.11
    Buy: $2808.48
    Sell: $2822.48 | profit: $32.83
    Sell: $2832.94 | profit: $61.49
    Buy: $2832.57
    Buy: $2824.23
    Sell: $2854.88 | profit: $105.95
    Buy: $2800.71
    Buy: $2798.36
    Sell: $2818.46 | profit: $75.39
    Buy: $2805.37
    Sell: $2815.44 | profit: $6.96
    Sell: $2834.40 | profit: $1.83
    Sell: $2867.19 | profit: $42.96
    Buy: $2867.24
    Sell: $2873.40 | profit: $72.69
    Sell: $2879.39 | profit: $81.03
    Sell: $2892.74 | profit: $87.37
    Sell: $2895.77 | profit: $28.53
    Buy: $2878.20
    Sell: $2888.21 | profit: $10.01
    Buy: $2888.32
    Sell: $2907.41 | profit: $19.09
    Buy: $2905.58
    Sell: $2907.06 | profit: $1.48
    Buy: $2900.45
    Sell: $2905.03 | profit: $4.58
    Buy: $2927.25
    Buy: $2926.17
    Sell: $2939.88 | profit: $12.63
    Sell: $2943.03 | profit: $16.86
    Buy: $2923.73
    Buy: $2917.52
    Sell: $2945.64 | profit: $21.91
    Buy: $2932.47
    Buy: $2884.05
    Buy: $2879.42
    Buy: $2870.72
    Sell: $2881.40 | profit: -$36.12
    Buy: $2811.87
    Sell: $2834.41 | profit: -$98.06
    Sell: $2850.96 | profit: -$33.09
    Sell: $2876.32 | profit: -$3.10
    Buy: $2859.53
    Buy: $2840.23
    Sell: $2864.36 | profit: -$6.36
    Buy: $2856.27
    Buy: $2822.24
    Sell: $2826.06 | profit: $14.19
    Buy: $2802.39
    Buy: $2783.02
    Sell: $2788.86 | profit: -$70.67
    Buy: $2752.06
    Buy: $2744.45
    Sell: $2803.27 | profit: -$36.96
    Sell: $2826.15 | profit: -$30.12
    Sell: $2843.49 | profit: $21.25
    Sell: $2873.34 | profit: $70.95
    Sell: $2886.73 | profit: $103.71
    Buy: $2885.72
    Buy: $2879.84
    Sell: $2891.64 | profit: $139.58
    Buy: $2886.98
    Sell: $2889.67 | profit: $145.22
    Sell: $2917.75 | profit: $32.03
    Sell: $2926.46 | profit: $46.62
    Sell: $2954.18 | profit: $67.20
    Buy: $2950.46
    Buy: $2945.35
    Buy: $2917.38
    Buy: $2913.78
    Sell: $2924.92 | profit: -$25.54
    Sell: $2941.76 | profit: -$3.59
    Sell: $2964.33 | profit: $46.95
    Sell: $2973.01 | profit: $59.23
    Buy: $2990.41
    Buy: $2975.95
    Sell: $2979.63 | profit: -$10.78
    Sell: $2993.07 | profit: $17.12
    Buy: $3004.04
    Buy: $2984.42
    Sell: $2995.11 | profit: -$8.93
    Buy: $2976.61
    Sell: $2985.03 | profit: $0.61
    Sell: $3005.47 | profit: $28.86
    Buy: $3003.67
    Sell: $3025.86 | profit: $22.19
    Buy: $3020.97
    Buy: $3013.18
    Buy: $2980.38
    Buy: $2953.56
    Buy: $2932.05
    Buy: $2844.74
    Sell: $2881.77 | profit: -$139.20
    Sell: $2883.98 | profit: -$129.20
    Sell: $2938.09 | profit: -$42.29
    Buy: $2918.65
    Buy: $2882.70
    Sell: $2926.32 | profit: -$27.24
    Buy: $2840.60
    Sell: $2847.60 | profit: -$84.45
    Sell: $2888.68 | profit: $43.94
    Sell: $2923.65 | profit: $5.00
    Buy: $2900.51
    Sell: $2924.43 | profit: $41.73
    Buy: $2922.95
    Buy: $2847.11
    Sell: $2878.38 | profit: $37.78
    Buy: $2869.16
    Sell: $2887.94 | profit: -$12.57
    Sell: $2924.58 | profit: $1.63
    Sell: $2926.46 | profit: $79.35
    Buy: $2906.27
    Sell: $2937.78 | profit: $68.62
    Sell: $2976.00 | profit: $69.73
    Buy: $2978.43
    Sell: $2979.39 | profit: $0.96
    Buy: $3007.39
    Buy: $2997.96
    Sell: $3005.70 | profit: -$1.69
    Sell: $3006.73 | profit: $8.77
    Buy: $3006.79
    Buy: $2992.07
    Buy: $2991.78
    Buy: $2966.60
    Sell: $2984.87 | profit: -$21.92
    Buy: $2977.62
    Buy: $2961.79
    Sell: $2976.74 | profit: -$15.33
    Buy: $2940.25
    Buy: $2887.61
    Sell: $2910.63 | profit: -$81.15
    Sell: $2952.01 | profit: -$14.59
    Buy: $2938.79
    Buy: $2893.06
    Sell: $2919.40 | profit: -$58.22
    Sell: $2938.13 | profit: -$23.66
    Sell: $2970.27 | profit: $30.02
    Buy: $2966.15
    Sell: $2995.68 | profit: $108.07
    Buy: $2989.69
    Sell: $2997.95 | profit: $59.16
    Buy: $2986.20
    Sell: $3006.72 | profit: $113.66
    Buy: $2995.99
    Sell: $3004.52 | profit: $38.37
    Sell: $3010.29 | profit: $20.60
    Sell: $3022.55 | profit: $36.35
    Sell: $3039.42 | profit: $43.43
    Buy: $3036.89
    Sell: $3046.77 | profit: $9.88
    Buy: $3037.56
    Sell: $3066.91 | profit: $29.35
    Buy: $3074.62
    Sell: $3076.78 | profit: $2.16
    Buy: $3087.01
    Sell: $3091.84 | profit: $4.83
    Buy: $3120.18
    Buy: $3108.46
    Buy: $3103.54
    Sell: $3110.29 | profit: -$9.89
    Sell: $3133.64 | profit: $25.18
    Sell: $3140.52 | profit: $36.98
    Buy: $3140.98
    Buy: $3113.87
    Buy: $3093.20
    Sell: $3112.76 | profit: -$28.22
    Sell: $3117.43 | profit: $3.56
    Sell: $3145.91 | profit: $52.71
    Buy: $3135.96
    Buy: $3132.52
    Sell: $3141.63 | profit: $5.67
    Sell: $3168.57 | profit: $36.05
    Buy: $3191.14
    Sell: $3205.37 | profit: $14.23
    Buy: $3223.38
    Sell: $3239.91 | profit: $16.53
    Buy: $3240.02
    Buy: $3221.29
    ------------------------------------------
    Total Profit: $1280.40
    ------------------------------------------



.. image:: output_45_1.png


Looking at the results above, our model resulted in an overall profit of
$1280, and we can say that our DQN agent performs quite well on the test
set. However, the performance of the model can be further improved by
optimizing the hyperparameters as discussed in the model tuning section
before. Also, given high complexity and low interpretability of the
model, ideally there should be more tests conducted on different time
periods before deploying the model for live trading.

**Conclusion**

We observed that we don’t have to decide the strategy or policy for
trading. The algorithm decides the policy by itself, and the overall
approach is much simpler and more principled than the supervised
learning-based approach.

The policy can be parameterized by a complex model, such as a deep
neural network, and we can learn policies that are more complex and
powerful than any rules a human trader.

We used the testing set to evaluate the model and found an overall
profit in the test set.
