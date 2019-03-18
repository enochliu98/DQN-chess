import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(

            self,

            n_actions,  # 行为个数

            n_features,  # 特征个数（？）

            learning_rate=0.01,  # α

            reward_decay=0.9,  # γ

            e_greedy=0.9,  # greedy

            replace_target_iter=300,  # 迭代的步数

            memory_size=500,  # 记忆空间，也就是replay中存储的

            batch_size=32,  # minibatch，用target_net得出来的

            e_greedy_increment=None,  #

            output_graph=False,

    ):
        self.n_actions = n_actions

        self.n_features = n_features #特征个数（）？

        self.lr = learning_rate

        self.gamma = reward_decay

        self.epsilon_max = e_greedy

        self.replace_target_iter = replace_target_iter

        self.memory_size = memory_size

        self.batch_size = batch_size

        self.epsilon_increment = e_greedy_increment

        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step

        self.learn_step_counter = 0

        # 初始化initialize zero memory [s, a, r, s_]

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # 程序包含两个网络，这是构建网络的方法，consist of [target_net, evaluate_net]

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # 用于讲eval的parameter复制给targetnet
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs

            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 500, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 400, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 500, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 400, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录一条 [s, a, r, s_] 记录
        transition = np.hstack((s, [a, r], s_))

        # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 替换过程

        self.memory_counter += 1


    def choose_action(self, observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)  # 随机选择
        return action


    # def learn(self):
    #     # 检查是否需要更新参数，每有一定步数要更新一次
    #
    #     if self.learn_step_counter % self.replace_target_iter == 0:
    #         self.sess.run(self.target_replace_op)
    #         print('\ntarget_params_replaced\n')
    #
    #     # 从 memory 中随机抽取 batch_size 这么多记忆
    #     if self.memory_counter > self.memory_size:
    #         sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    #     else:
    #         sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
    #     batch_memory = self.memory[sample_index, :]
    #
    #     # 通过神经网络获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
    #     q_next, q_eval = self.sess.run(
    #         [self.q_next, self.q_eval],
    #         feed_dict={
    #             self.s_: batch_memory[:, -self.n_features:],
    #             self.s: batch_memory[:, :self.n_features]
    #         })
    #
    #     # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
    #     # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
    #     # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
    #     # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
    #     # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
    #     # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
    #     # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.
    #
    #     # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
    #     # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
    #     # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
    #     # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
    #     # 具体在下面还有一个举例说明.
    #
    #     q_target = q_eval.copy()
    #     #print(np.shape(q_target))
    #     batch_index = np.arange(self.batch_size, dtype=np.int32)
    #     eval_act_index = batch_memory[:, self.n_features].astype(int)
    #     reward = batch_memory[:, self.n_features + 1]
    #
    #     q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
    #
    #     # 训练 eval_net
    #     _, self.cost = self.sess.run([self._train_op, self.loss],
    #                              feed_dict={self.s: batch_memory[:, :self.n_features],
    #                                         self.q_target: q_target})
    #     self.cost_his.append(self.cost)  # 记录 cost 误差
    #
    #     # 逐渐增加 epsilon, 降低行为的随机性
    #     self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
    #     self.learn_step_counter += 1

    def learn(self):

        # check to replace target parameters

        if self.learn_step_counter % self.replace_target_iter == 0:

            self.sess.run(self.target_replace_op)

            print('\ntarget_params_replaced\n')



        # sample batch memory from all memory

        if self.memory_counter > self.memory_size:

            sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        else:

            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]



        _, cost = self.sess.run(

            [self._train_op, self.loss],

            feed_dict={

                self.s: batch_memory[:, :self.n_features],

                self.a: batch_memory[:, self.n_features],

                self.r: batch_memory[:, self.n_features + 1],

                self.s_: batch_memory[:, -self.n_features:],

            })



        self.cost_his.append(cost)



        # increasing epsilon

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1



if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)