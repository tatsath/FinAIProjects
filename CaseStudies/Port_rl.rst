.. _Port_rl:



Reinforcement Learning for Portfolio Allocation
===============================================

In this case study, similar to Case Study 1 of this chapter, we will use
the Reinforcement Learning models to come up with a policy for optimal
portfolio allocation among a set of cryptocurrencies.

Content
-------

-  `1. Problem Definition <#0>`__
-  `2. Getting Started - Load Libraries and Dataset <#1>`__

   -  `2.1. Load Libraries <#1.1>`__
   -  `2.2. Load Dataset <#1.2>`__

-  `3. Exploratory Data Analysis <#2>`__

   -  `3.1 Descriptive Statistics <#2.1>`__
   -  `3.2. Data Visualisation <#2.2>`__

-  `4.Evaluate Algorithms and Models <#4>`__

   -  `4.1. Defining the Environment <#4.1>`__
   -  `4.2. Agent Script <#4.2>`__
   -  `4.3. Training the model <#4.3>`__

-  `5.Testing the Model <#5>`__

 # 1. Problem Definition

In the reinforcement learning based framework defined for this problem,
the algorithm determines the optimal portfolio allocation depending upon
the current state of the portfolio of instruments.

The algorithm is trained using Deep QLearning framework and the
components of the reinforcement learning environment are:

-  Agent: Portfolio manager, robo advisor or an individual.
-  Action: Assignment and rebalancing the portfolio weights. The DQN
   model provides the Q-values which is further converted into portfolio
   weights.

-  Reward function: Sharpe ratio, which consists of the standard
   deviation as the risk assessment measure is used reward function.

-  State: The state is the correlation matrix of the instruments based
   on a specific time window. The correlation matrix is a suitable state
   variable for the portfolio allocation, as it contains the information
   about the relationships between different instruments and can be
   useful in performing portfolio allocation.

-  Environment: Cryptocurrency exchange.

The data of cryptocurrencies that we will be using for this case study
is obtained from the Kaggle platform and contains the daily prices of
the cryptocurrencies during the period of 2018. The data contains some
of the most liquid cryptocurrencies such as Bitcoin, Ethereum, Ripple,
Litecoin and Dash.

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

    from keras.layers import Input, Dense, Flatten, Dropout
    from keras.models import Model
    from keras.regularizers import l2

    import numpy as np
    import pandas as pd

    import random
    from collections import deque
    import matplotlib.pylab as plt

.. code:: ipython3

    #Diable the warnings
    import warnings
    warnings.filterwarnings('ignore')

 ## 2.2. Loading the Data

.. code:: ipython3

    #The data already obtained from yahoo finance is imported.
    dataset = read_csv('data/crypto_portfolio.csv',index_col=0)

 # 3. Exploratory Data Analysis

 ## 3.1. Descriptive Statistics

.. code:: ipython3

    # shape
    dataset.shape




.. parsed-literal::

    (375, 15)



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
          <th>ADA</th>
          <th>BCH</th>
          <th>BNB</th>
          <th>BTC</th>
          <th>DASH</th>
          <th>EOS</th>
          <th>ETH</th>
          <th>IOT</th>
          <th>LINK</th>
          <th>LTC</th>
          <th>TRX</th>
          <th>USDT</th>
          <th>XLM</th>
          <th>XMR</th>
          <th>XRP</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
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
          <th>2018-01-01</th>
          <td>0.7022</td>
          <td>2319.120117</td>
          <td>8.480</td>
          <td>13444.879883</td>
          <td>1019.419983</td>
          <td>7.64</td>
          <td>756.200012</td>
          <td>3.90</td>
          <td>0.7199</td>
          <td>224.339996</td>
          <td>0.05078</td>
          <td>1.01</td>
          <td>0.4840</td>
          <td>338.170013</td>
          <td>2.05</td>
        </tr>
        <tr>
          <th>2018-01-02</th>
          <td>0.7620</td>
          <td>2555.489990</td>
          <td>8.749</td>
          <td>14754.129883</td>
          <td>1162.469971</td>
          <td>8.30</td>
          <td>861.969971</td>
          <td>3.98</td>
          <td>0.6650</td>
          <td>251.809998</td>
          <td>0.07834</td>
          <td>1.02</td>
          <td>0.5560</td>
          <td>364.440002</td>
          <td>2.19</td>
        </tr>
        <tr>
          <th>2018-01-03</th>
          <td>1.1000</td>
          <td>2557.520020</td>
          <td>9.488</td>
          <td>15156.620117</td>
          <td>1129.890015</td>
          <td>9.43</td>
          <td>941.099976</td>
          <td>4.13</td>
          <td>0.6790</td>
          <td>244.630005</td>
          <td>0.09430</td>
          <td>1.01</td>
          <td>0.8848</td>
          <td>385.820007</td>
          <td>2.73</td>
        </tr>
        <tr>
          <th>2018-01-04</th>
          <td>1.1300</td>
          <td>2355.780029</td>
          <td>9.143</td>
          <td>15180.080078</td>
          <td>1120.119995</td>
          <td>9.47</td>
          <td>944.830017</td>
          <td>4.10</td>
          <td>0.9694</td>
          <td>238.300003</td>
          <td>0.21010</td>
          <td>1.02</td>
          <td>0.6950</td>
          <td>372.230011</td>
          <td>2.73</td>
        </tr>
        <tr>
          <th>2018-01-05</th>
          <td>1.0100</td>
          <td>2390.040039</td>
          <td>14.850</td>
          <td>16954.779297</td>
          <td>1080.880005</td>
          <td>9.29</td>
          <td>967.130005</td>
          <td>3.76</td>
          <td>0.9669</td>
          <td>244.509995</td>
          <td>0.22400</td>
          <td>1.01</td>
          <td>0.6400</td>
          <td>357.299988</td>
          <td>2.51</td>
        </tr>
      </tbody>
    </table>
    </div>



The data is the historical data of several Cryptocurrencies

 # 4. Evaluate Algorithms and Models

We will look at the following Scripts :

1. Creating Environment
2. Helper Functions
3. Training Agents

 ## 4.1. Cryptocurrency environment

We introduce a simulation environment class “CryptoEnvironment”, where
we create a working environment for cryptocurrencies. This class has
following key functions:

-  Function “getState: This function returns the state, which is the
   correlation matrix of the instruments based on a lookback period. The
   function also returns the historical return or raw historical data as
   the state depending on is_cov_matrix or is_raw_time_series flag.
-  Function “getReward: This function returns the reward, which is sharp
   ratio of the portfolio, given the portfolio weight and lookback
   period.

.. code:: ipython3

    import numpy as np
    import pandas as pd

    from IPython.core.debugger import set_trace

    #define a function portfolio
    def portfolio(returns, weights):
        weights = np.array(weights)
        rets = returns.mean() * 252
        covs = returns.cov() * 252
        P_ret = np.sum(rets * weights)
        P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
        P_sharpe = P_ret / P_vol
        return np.array([P_ret, P_vol, P_sharpe])


    class CryptoEnvironment:

        def __init__(self, prices = './data/crypto_portfolio.csv', capital = 1e6):
            self.prices = prices
            self.capital = capital
            self.data = self.load_data()

        def load_data(self):
            data =  pd.read_csv(self.prices)
            try:
                data.index = data['Date']
                data = data.drop(columns = ['Date'])
            except:
                data.index = data['date']
                data = data.drop(columns = ['date'])
            return data

        def preprocess_state(self, state):
            return state

        def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):

            assert lookback <= t

            decision_making_state = self.data.iloc[t-lookback:t]
            decision_making_state = decision_making_state.pct_change().dropna()
            #set_trace()
            if is_cov_matrix:
                x = decision_making_state.cov()
                return x
            else:
                if is_raw_time_series:
                    decision_making_state = self.data.iloc[t-lookback:t]
                return self.preprocess_state(decision_making_state)

        def get_reward(self, action, action_t, reward_t, alpha = 0.01):

            def local_portfolio(returns, weights):
                weights = np.array(weights)
                rets = returns.mean() # * 252
                covs = returns.cov() # * 252
                P_ret = np.sum(rets * weights)
                P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
                P_sharpe = P_ret / P_vol
                return np.array([P_ret, P_vol, P_sharpe])

            data_period = self.data[action_t:reward_t]
            weights = action
            returns = data_period.pct_change().dropna()

            sharpe = local_portfolio(returns, weights)[-1]
            sharpe = np.array([sharpe] * len(self.data.columns))
            rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]

            return np.dot(returns, weights), sharpe



 ## 4.2. Agent Script

In this section, we will train an agent that will perform reinforcement
learning based on the actor and critic networks. We will perform the
following steps to achieve this: \* Create an agent class whose initial
function takes in the batch size, state size, and an evaluation Boolean
function, to check whether the training is ongoing. \* In the agent
class, create the following methods: \* Create a Replay function that
adds, samples, and evaluates a buffer. \* Add a new experience to the
replay buffer memory \* Randomly sample a batch of experienced tuples
from the memory. In the following function, we randomly sample states
from a memory buffer. We do this so that the states that we feed to the
model are not temporally correlated. This will reduce overfitting: \*
Return the current size of the buffer memory \* The number of actions
are defined as 3: sit, buy, sell \* Define the replay memory size \*
Reward function is return

.. code:: ipython3

    class Agent:

        def __init__(
                         self,
                         portfolio_size,
                         is_eval = False,
                         allow_short = True,
                     ):

            self.portfolio_size = portfolio_size
            self.allow_short = allow_short
            self.input_shape = (portfolio_size, portfolio_size, )
            self.action_size = 3 # sit, buy, sell

            self.memory4replay = []
            self.is_eval = is_eval

            self.alpha = 0.5
            self.gamma = 0.95
            self.epsilon = 1
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.99

            self.model = self._model()

        def _model(self):

            inputs = Input(shape=self.input_shape)
            x = Flatten()(inputs)
            x = Dense(100, activation='elu')(x)
            x = Dropout(0.5)(x)
            x = Dense(50, activation='elu')(x)
            x = Dropout(0.5)(x)

            predictions = []
            for i in range(self.portfolio_size):
                asset_dense = Dense(self.action_size, activation='linear')(x)
                predictions.append(asset_dense)

            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='adam', loss='mse')
            return model

        def nn_pred_to_weights(self, pred, allow_short = False):

            weights = np.zeros(len(pred))
            raw_weights = np.argmax(pred, axis=-1)

            saved_min = None

            for e, r in enumerate(raw_weights):
                if r == 0: # sit
                    weights[e] = 0
                elif r == 1: # buy
                    weights[e] = np.abs(pred[e][0][r])
                else:
                    weights[e] = -np.abs(pred[e][0][r])
            #sum of absolute values in short is allowed
            if not allow_short:
                weights += np.abs(np.min(weights))
                saved_min = np.abs(np.min(weights))
                saved_sum = np.sum(weights)
            else:
                saved_sum = np.sum(np.abs(weights))

            weights /= saved_sum
            return weights, saved_min, saved_sum
        #return the action based on the state, uses the NN function
        def act(self, state):

            if not self.is_eval and random.random() <= self.epsilon:
                w = np.random.normal(0, 1, size = (self.portfolio_size, ))

                saved_min = None

                if not self.allow_short:
                    w += np.abs(np.min(w))
                    saved_min = np.abs(np.min(w))

                saved_sum = np.sum(w)
                w /= saved_sum
                return w , saved_min, saved_sum

            pred = self.model.predict(np.expand_dims(state.values, 0))
            return self.nn_pred_to_weights(pred, self.allow_short)

        def expReplay(self, batch_size):

            def weights_to_nn_preds_with_reward(action_weights,
                                                reward,
                                                Q_star = np.zeros((self.portfolio_size, self.action_size))):

                Q = np.zeros((self.portfolio_size, self.action_size))
                for i in range(self.portfolio_size):
                    if action_weights[i] == 0:
                        Q[i][0] = reward[i] + self.gamma * np.max(Q_star[i][0])
                    elif action_weights[i] > 0:
                        Q[i][1] = reward[i] + self.gamma * np.max(Q_star[i][1])
                    else:
                        Q[i][2] = reward[i] + self.gamma * np.max(Q_star[i][2])
                return Q

            def restore_Q_from_weights_and_stats(action):
                action_weights, action_min, action_sum = action[0], action[1], action[2]
                action_weights = action_weights * action_sum
                if action_min != None:
                    action_weights = action_weights - action_min
                return action_weights

            for (s, s_, action, reward, done) in self.memory4replay:

                action_weights = restore_Q_from_weights_and_stats(action)
                #Reward =reward if not in the terminal state.
                Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward)
                s, s_ = s.values, s_.values

                if not done:
                    # reward + gamma * Q^*(s_, a_)
                    Q_star = self.model.predict(np.expand_dims(s_, 0))
                    Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward, np.squeeze(Q_star))

                Q_learned_value = [xi.reshape(1, -1) for xi in Q_learned_value]
                Q_current_value = self.model.predict(np.expand_dims(s, 0))
                Q = [np.add(a * (1-self.alpha), q * self.alpha) for a, q in zip(Q_current_value, Q_learned_value)]

                # update current Q function with new optimal value
                self.model.fit(np.expand_dims(s, 0), Q, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

 ## 4.3. Training the data

In this step we train the algorithm. In order to do that, we first
initialize the “Agent” class and “CryptoEnvironment” class.

.. code:: ipython3

    N_ASSETS = 15 #53
    agent = Agent(N_ASSETS)
    env = CryptoEnvironment()

.. code:: ipython3

    window_size = 180
    episode_count = 50
    batch_size = 32
    rebalance_period = 90 #every 90 days weight change

.. code:: ipython3

    data_length = len(env.data)
    data_length




.. parsed-literal::

    375



.. code:: ipython3

    np.random.randint(window_size+1, data_length-window_size-1)




.. parsed-literal::

    181



.. code:: ipython3

    for e in range(episode_count):

        agent.is_eval = False
        data_length = len(env.data)

        returns_history = []
        returns_history_equal = []

        rewards_history = []
        equal_rewards = []

        actions_to_show = []

        print("Episode " + str(e) + "/" + str(episode_count), 'epsilon', agent.epsilon)

        s = env.get_state(np.random.randint(window_size+1, data_length-window_size-1), window_size)
        total_profit = 0

        for t in range(window_size, data_length, rebalance_period):
            date1 = t-rebalance_period
            #correlation from 90-180 days
            s_ = env.get_state(t, window_size)
            action = agent.act(s_)

            actions_to_show.append(action[0])

            weighted_returns, reward = env.get_reward(action[0], date1, t)
            weighted_returns_equal, reward_equal = env.get_reward(
                np.ones(agent.portfolio_size) / agent.portfolio_size, date1, t)

            rewards_history.append(reward)
            equal_rewards.append(reward_equal)
            returns_history.extend(weighted_returns)
            returns_history_equal.extend(weighted_returns_equal)

            done = True if t == data_length else False
            agent.memory4replay.append((s, s_, action, reward, done))

            if len(agent.memory4replay) >= batch_size:
                agent.expReplay(batch_size)
                agent.memory4replay = []

            s = s_

        rl_result = np.array(returns_history).cumsum()
        equal_result = np.array(returns_history_equal).cumsum()

        plt.figure(figsize = (12, 2))
        plt.plot(rl_result, color = 'black', ls = '-')
        plt.plot(equal_result, color = 'grey', ls = '--')
        plt.show()

        plt.figure(figsize = (12, 2))
        for a in actions_to_show:
            plt.bar(np.arange(N_ASSETS), a, color = 'grey', alpha = 0.25)
            plt.xticks(np.arange(N_ASSETS), env.data.columns, rotation='vertical')
        plt.show()



.. parsed-literal::

    Episode 0/50 epsilon 1



.. image:: output_29_1.png



.. image:: output_29_2.png


.. parsed-literal::

    Episode 1/50 epsilon 1



.. image:: output_29_4.png



.. image:: output_29_5.png


.. parsed-literal::

    Episode 2/50 epsilon 1



.. image:: output_29_7.png



.. image:: output_29_8.png


.. parsed-literal::

    Episode 3/50 epsilon 1



.. image:: output_29_10.png



.. image:: output_29_11.png


.. parsed-literal::

    Episode 4/50 epsilon 1



.. image:: output_29_13.png



.. image:: output_29_14.png


.. parsed-literal::

    Episode 5/50 epsilon 1



.. image:: output_29_16.png



.. image:: output_29_17.png


.. parsed-literal::

    Episode 6/50 epsilon 1



.. image:: output_29_19.png



.. image:: output_29_20.png


.. parsed-literal::

    Episode 7/50 epsilon 1



.. image:: output_29_22.png



.. image:: output_29_23.png


.. parsed-literal::

    Episode 8/50 epsilon 1



.. image:: output_29_25.png



.. image:: output_29_26.png


.. parsed-literal::

    Episode 9/50 epsilon 1



.. image:: output_29_28.png



.. image:: output_29_29.png


.. parsed-literal::

    Episode 10/50 epsilon 1



.. image:: output_29_31.png



.. image:: output_29_32.png


.. parsed-literal::

    Episode 11/50 epsilon 0.99



.. image:: output_29_34.png



.. image:: output_29_35.png


.. parsed-literal::

    Episode 12/50 epsilon 0.99



.. image:: output_29_37.png



.. image:: output_29_38.png


.. parsed-literal::

    Episode 13/50 epsilon 0.99



.. image:: output_29_40.png



.. image:: output_29_41.png


.. parsed-literal::

    Episode 14/50 epsilon 0.99



.. image:: output_29_43.png



.. image:: output_29_44.png


.. parsed-literal::

    Episode 15/50 epsilon 0.99



.. image:: output_29_46.png



.. image:: output_29_47.png


.. parsed-literal::

    Episode 16/50 epsilon 0.99



.. image:: output_29_49.png



.. image:: output_29_50.png


.. parsed-literal::

    Episode 17/50 epsilon 0.99



.. image:: output_29_52.png



.. image:: output_29_53.png


.. parsed-literal::

    Episode 18/50 epsilon 0.99



.. image:: output_29_55.png



.. image:: output_29_56.png


.. parsed-literal::

    Episode 19/50 epsilon 0.99



.. image:: output_29_58.png



.. image:: output_29_59.png


.. parsed-literal::

    Episode 20/50 epsilon 0.99



.. image:: output_29_61.png



.. image:: output_29_62.png


.. parsed-literal::

    Episode 21/50 epsilon 0.99



.. image:: output_29_64.png



.. image:: output_29_65.png


.. parsed-literal::

    Episode 22/50 epsilon 0.9801



.. image:: output_29_67.png



.. image:: output_29_68.png


.. parsed-literal::

    Episode 23/50 epsilon 0.9801



.. image:: output_29_70.png



.. image:: output_29_71.png


.. parsed-literal::

    Episode 24/50 epsilon 0.9801



.. image:: output_29_73.png



.. image:: output_29_74.png


.. parsed-literal::

    Episode 25/50 epsilon 0.9801



.. image:: output_29_76.png



.. image:: output_29_77.png


.. parsed-literal::

    Episode 26/50 epsilon 0.9801



.. image:: output_29_79.png



.. image:: output_29_80.png


.. parsed-literal::

    Episode 27/50 epsilon 0.9801



.. image:: output_29_82.png



.. image:: output_29_83.png


.. parsed-literal::

    Episode 28/50 epsilon 0.9801



.. image:: output_29_85.png



.. image:: output_29_86.png


.. parsed-literal::

    Episode 29/50 epsilon 0.9801



.. image:: output_29_88.png



.. image:: output_29_89.png


.. parsed-literal::

    Episode 30/50 epsilon 0.9801



.. image:: output_29_91.png



.. image:: output_29_92.png


.. parsed-literal::

    Episode 31/50 epsilon 0.9801



.. image:: output_29_94.png



.. image:: output_29_95.png


.. parsed-literal::

    Episode 32/50 epsilon 0.9702989999999999



.. image:: output_29_97.png



.. image:: output_29_98.png


.. parsed-literal::

    Episode 33/50 epsilon 0.9702989999999999



.. image:: output_29_100.png



.. image:: output_29_101.png


.. parsed-literal::

    Episode 34/50 epsilon 0.9702989999999999



.. image:: output_29_103.png



.. image:: output_29_104.png


.. parsed-literal::

    Episode 35/50 epsilon 0.9702989999999999



.. image:: output_29_106.png



.. image:: output_29_107.png


.. parsed-literal::

    Episode 36/50 epsilon 0.9702989999999999



.. image:: output_29_109.png



.. image:: output_29_110.png


.. parsed-literal::

    Episode 37/50 epsilon 0.9702989999999999



.. image:: output_29_112.png



.. image:: output_29_113.png


.. parsed-literal::

    Episode 38/50 epsilon 0.9702989999999999



.. image:: output_29_115.png



.. image:: output_29_116.png


.. parsed-literal::

    Episode 39/50 epsilon 0.9702989999999999



.. image:: output_29_118.png



.. image:: output_29_119.png


.. parsed-literal::

    Episode 40/50 epsilon 0.9702989999999999



.. image:: output_29_121.png



.. image:: output_29_122.png


.. parsed-literal::

    Episode 41/50 epsilon 0.9702989999999999



.. image:: output_29_124.png



.. image:: output_29_125.png


.. parsed-literal::

    Episode 42/50 epsilon 0.9702989999999999



.. image:: output_29_127.png



.. image:: output_29_128.png


.. parsed-literal::

    Episode 43/50 epsilon 0.96059601



.. image:: output_29_130.png



.. image:: output_29_131.png


.. parsed-literal::

    Episode 44/50 epsilon 0.96059601



.. image:: output_29_133.png



.. image:: output_29_134.png


.. parsed-literal::

    Episode 45/50 epsilon 0.96059601



.. image:: output_29_136.png



.. image:: output_29_137.png


.. parsed-literal::

    Episode 46/50 epsilon 0.96059601



.. image:: output_29_139.png



.. image:: output_29_140.png


.. parsed-literal::

    Episode 47/50 epsilon 0.96059601



.. image:: output_29_142.png



.. image:: output_29_143.png


.. parsed-literal::

    Episode 48/50 epsilon 0.96059601



.. image:: output_29_145.png



.. image:: output_29_146.png


.. parsed-literal::

    Episode 49/50 epsilon 0.96059601



.. image:: output_29_148.png



.. image:: output_29_149.png


The charts shown above show the details of the portfolio allocation of
all the episodes.

 # 5. Testing the Data

After training the data, it is tested it against the test dataset.

.. code:: ipython3

    agent.is_eval = True

    actions_equal, actions_rl = [], []
    result_equal, result_rl = [], []

    for t in range(window_size, len(env.data), rebalance_period):

        date1 = t-rebalance_period
        s_ = env.get_state(t, window_size)
        action = agent.act(s_)

        weighted_returns, reward = env.get_reward(action[0], date1, t)
        weighted_returns_equal, reward_equal = env.get_reward(
            np.ones(agent.portfolio_size) / agent.portfolio_size, date1, t)

        result_equal.append(weighted_returns_equal.tolist())
        actions_equal.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

        result_rl.append(weighted_returns.tolist())
        actions_rl.append(action[0])

.. code:: ipython3

    result_equal_vis = [item for sublist in result_equal for item in sublist]
    result_rl_vis = [item for sublist in result_rl for item in sublist]

.. code:: ipython3

    plt.figure()
    plt.plot(np.array(result_equal_vis).cumsum(), label = 'Benchmark', color = 'grey',ls = '--')
    plt.plot(np.array(result_rl_vis).cumsum(), label = 'Deep RL portfolio', color = 'black',ls = '-')
    plt.show()



.. image:: output_35_0.png


.. code:: ipython3

    #Plotting the data
    import matplotlib
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='red')

    N = len(np.array([item for sublist in result_equal for item in sublist]).cumsum())

    for i in range(0, len(actions_rl)):
        current_range = np.arange(0, N)
        current_ts = np.zeros(N)
        current_ts2 = np.zeros(N)

        ts_benchmark = np.array([item for sublist in result_equal[:i+1] for item in sublist]).cumsum()
        ts_target = np.array([item for sublist in result_rl[:i+1] for item in sublist]).cumsum()

        t = len(ts_benchmark)
        current_ts[:t] = ts_benchmark
        current_ts2[:t] = ts_target

        current_ts[current_ts == 0] = ts_benchmark[-1]
        current_ts2[current_ts2 == 0] = ts_target[-1]

        plt.figure(figsize = (12, 10))

        plt.subplot(2, 1, 1)
        plt.bar(np.arange(N_ASSETS), actions_rl[i], color = 'grey')
        plt.xticks(np.arange(N_ASSETS), env.data.columns, rotation='vertical')

        plt.subplot(2, 1, 2)
        plt.colormaps = current_cmap
        plt.plot(current_range[:t], current_ts[:t], color = 'black', label = 'Benchmark')
        plt.plot(current_range[:t], current_ts2[:t], color = 'red', label = 'Deep RL portfolio')
        plt.plot(current_range[t:], current_ts[t:], ls = '--', lw = .1, color = 'black')
        plt.autoscale(False)
        plt.ylim([-1, 1])
        plt.legend()



.. image:: output_36_0.png



.. image:: output_36_1.png



.. image:: output_36_2.png


.. code:: ipython3

    import statsmodels.api as sm
    from statsmodels import regression
    def sharpe(R):
        r = np.diff(R)
        sr = r.mean()/r.std() * np.sqrt(252)
        return sr

    def print_stats(result, benchmark):

        sharpe_ratio = sharpe(np.array(result).cumsum())
        returns = np.mean(np.array(result))
        volatility = np.std(np.array(result))

        X = benchmark
        y = result
        x = sm.add_constant(X)
        model = regression.linear_model.OLS(y, x).fit()
        alpha = model.params[0]
        beta = model.params[1]

        return np.round(np.array([returns, volatility, sharpe_ratio, alpha, beta]), 4).tolist()

.. code:: ipython3

    print('EQUAL', print_stats(result_equal_vis, result_equal_vis))
    print('RL AGENT', print_stats(result_rl_vis, result_equal_vis))


.. parsed-literal::

    EQUAL [-0.0013, 0.0468, -0.5016, 0.0, 1.0]
    RL AGENT [0.0004, 0.0231, 0.4445, 0.0002, -0.1202]


RL portfolio has a higher return, higher sharp, lower volatility, higher
alpha and negative correlation with the benchmark.

**Conclusion**

The idea in this case study was to go beyond classical Markowitz
efficient frontier and directly learn the policy of changing the weights
dynamically in the continuously changing market.

We set up a standardized working environ‐ ment(“gym”) for
cryptocurrencies to facilitate the training. The model starts to learn
over a period of time, discovers the strategy and starts to exploit it.
we used the testing set to evaluate the model and found an overall
profit in the test set.

Overall, the framework provided in this case study can enable financial
practitioners to perform portfolio allocation and rebalancing with a
very flexible and automated approach and can prove to be immensely
useful, specifically for robo-advisors
