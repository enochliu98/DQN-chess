from env import cheese

from DeepQNetwork import DeepQNetwork





def run_maze():

    step = 0

    for episode in range(300):

        # initial observation

        observation = env.reset()



        while True:

            # RL choose action based on observation
            # 运用训练好的神经网络的出来预测的值，然后选一个最大的，将它作为现在这个状态的动作选择
            action = RL.choose_action(observation)



            # RL take action and get next observation and reward
            # 通过环境获取在当前状态下选取最好动作，下一步的观察结果，奖励和是否中止
            observation_, reward, done = env.step(action,observation)


            #存储当前的状态转移序列
            RL.store_transition(observation, action, reward, observation_)


            #根据步数判断是否进行学习
            #开始的时候，进行数据存储不进行学习，等到存到一定地步再进行学习
            if (step > 200) and (step % 5 == 0):

                RL.learn()



            # swap observation

            observation = observation_



            # break while loop when end of this episode

            if done:

                break

            step += 1



    # end of game

    print('game over')

    env.destroy()





if __name__ == "__main__":

    # maze game

    env = cheese()

    RL = DeepQNetwork(env.n_actions, env.n_features,

                      learning_rate=0.01,

                      reward_decay=0.9,

                      e_greedy=0.9,

                      replace_target_iter=200,

                      memory_size=2000,

                      # output_graph=True

                      )
    run_maze()
    ########这里是进行训练的次数调整
