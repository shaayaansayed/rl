import gym
import tensorflow as tf
import numpy as np 
import random
import argparse

class Network() :
    
    def __init__(self, num_actions, scope='Network', summary_dir=None) :
        self.num_actions = num_actions 
        
        self.fw = None
        
        with tf.name_scope(scope) :
            if summary_dir :
                self.fw = tf.summary.FileWriter(summary_dir)

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.build_network()
        
    def build_network(self) :
        self.img = tf.placeholder(tf.float32, shape=[None, 84, 84, 1], name="X")
        self.action_ix = tf.placeholder(tf.int32, shape=[None], name='action_ix')
        self.td_target = tf.placeholder(tf.float32, shape=[None], name='Y')

        conv1 = tf.layers.conv2d(inputs=self.img, filters=32, kernel_size=[8, 8], 
                                 strides=(4, 4), padding='SAME', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], 
                                 strides=(2, 2), padding='SAME', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], 
                                 strides=(1, 1), padding='SAME', activation=tf.nn.relu)

        conv3_flat = tf.contrib.layers.flatten(conv3)
        dense = tf.layers.dense(conv3_flat, units=1024)
        self.action_vals = tf.layers.dense(dense, units=self.num_actions)

        batch_size = tf.shape(self.img)[0]
        gather_ix = tf.range(batch_size) * self.num_actions + self.action_ix
        pred_action_val = tf.gather(tf.reshape(self.action_vals, [-1]), gather_ix)
        
        self.loss = tf.reduce_mean(tf.squared_difference(self.td_target, pred_action_val))
        optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()
    
    def predict(self, sess, obs) :
        net_input = obs 
        if obs.ndim == 3 :
            net_input = np.expand_dims(obs, axis=0)
            
        action_values = sess.run(self.action_vals, {self.img:net_input})
        return action_values

    def update(self, sess, obs_t, A_ix, targets) :
        global_step, loss, _, summaries = sess.run([self.global_step, self.loss, self.train_op, self.summary_op], \
                                      {self.img: obs_t, self.action_ix: A_ix, self.td_target: targets})

        if self.fw :
            self.fw.add_summary(summaries, global_step)

        return loss 

class ObsProcessor() :

    def __init__(self) :
        self.raw_imgs = tf.placeholder(tf.int8, shape=[210, 160, 3], name="raw_imgs")
        grayscale = tf.image.rgb_to_grayscale(self.raw_imgs)
        resized = tf.image.resize_images(
                        grayscale, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.output = tf.cast(resized, tf.float32)

    def preprocess(self, sess, obs) :
        return sess.run(self.output, {self.raw_imgs: obs})

def init_greedy_pi(Qnet, num_actions, ep) :
    def pi(sess, obs) :

        Qs = Qnet.predict(sess, obs)
        greedy_action = np.argmax(Qs)

        dist = np.ones(num_actions) * ep/num_actions
        dist[greedy_action] += 1 - ep

        return np.random.choice(num_actions, p=dist)
    return pi


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

def run_Q_learning(args) :

    env = gym.make('Breakout-v0')

    num_actions = env.action_space.n

    processor = ObsProcessor()
    Q_net = Network(num_actions, scope='Q', summary_dir=args.summary_dir)
    target_net = Network(num_actions, scope='td_target')
        
    pi = init_greedy_pi(Q_net, num_actions, args.ep)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # populate initial replay memory

    replay_memory = []

    while len(replay_memory) < args.init_replay_size :
        
        done = False
        obs_t = env.reset()
        obs_t = processor.preprocess(sess, obs_t)
        while not done :
            A_t = pi(sess, obs_t)
            obs_tp1, R_tp1, done, info = env.step(A_t)
            replay_memory.append([obs_t, A_t, R_tp1])
            obs_t = processor.preprocess(sess, obs_tp1)

    # main loop 

    iter_ix = 0 
    for ix in range(args.num_episodes) :
        
        ep_length = 0
        ep_reward = 0 

        done = False
        obs_t = env.reset()
        obs_t = processor.preprocess(sess, obs_t)
        while not done :
            iter_ix = iter_ix + 1

            if iter_ix % args.update_target_net_every == 0 :
                copy_model_parameters(sess, Q_net, target_net)

            A_t = pi(sess, obs_t)
            obs_tp1, R_tp1, done, info = env.step(A_t)
            replay_memory.append([obs_t, A_t, R_tp1])

            # sample minibatch from replay_memory
            replay_sample = random.sample(replay_memory, args.batch_size)
            obs_batch, action_batch, reward_batch = map(np.array, zip(*replay_sample))

            # compute TD targets using target network 
            target_Qs = target_net.predict(sess, obs_batch)
            max_Qs = np.amax(target_Qs, axis=0)
            TD_targets = reward_batch + max_Qs

            # update Q Network 
            loss = Q_net.update(sess, obs_batch, action_batch, TD_targets)

            obs_t = processor.preprocess(sess, obs_tp1)

            print('{} iter : {}'.format(iter_ix, loss))

            ep_reward += R_tp1
            ep_length += 1


        ep_summary = tf.Summary(value =[tf.Summary.Value(tag='ep_reward', simple_value=ep_reward), 
                                         tf.Summary.Value(tag='ep_length', simple_value=ep_length)])
        Q_net.fw.add_summary(ep_summary, global_step=iter_ix)
        Q_net.fw.flush()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='arguments for dqn')

    parser.add_argument('-batch_size', default=4, dest='batch_size', type=int)
    parser.add_argument('-init_replay_size', default=64, dest='init_replay_size', type=int)
    parser.add_argument('-ep', default=.1, dest='ep', type=float)
    parser.add_argument('-summary_dir', default="./summary/", dest='summary_dir', type=str)
    parser.add_argument('-num_episodes', default=16, dest='num_episodes', type=int)
    parser.add_argument('-update_target_net_every', default=10000, dest='update_target_net_every', type=int)

    args = parser.parse_args()

    run_Q_learning(args)
