import gym
import tensorflow as tf
import numpy as np 
import random
import argparse

from StringIO import StringIO
import matplotlib.pyplot as plt

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           intializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           padding='VALID',
           name='conv2d') :

    with tf.variable_scope(name) :
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv2d = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))

        out = tf.nn.bias_add(conv, b, data_format)

        if activation_fn :
            out = activation_fn(out)
        return out, w, b

def linear(x,
           output_dim,
           stddev=0.02,
           bias_start=0.0,
           activation_fn=tf.nn.relu,
           name='linear') :
    
    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn:
            out = activation_fn(out)
        return out, w, b

class Network() :

    def __init__(self, sess, num_actions, img_size) :
        self.sess = sess 
        self.num_actions = num_actions 
        self.img_size = img_size

    def build_dqn(self) :

        self.pred_w = {}
        with tf.variable_scope('prediction') :

            self.obs = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], 1], name='obs')

            self.l1, self.pred_w['l1_w'], self.pred_w['l1_b'] = conv2d(self.obs, output_dim=32, kernel_size=[8, 8], stride=[4, 4], name='l1')
            self.l2, self.pred_w['l2_w'], self.pred_w['l2_b'] = conv2d(self.l1, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l2')
            self.l3, self.pred_w['l3_w'], self.pred_w['l3_b'] = conv2d(self.l2, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l3')

            flat_size = reduce(lambda x,y : x*y, tf.get_shape(self.l3).as_list()[1:])
            self.l3_flat = tf.reshape(self.l3, [-1, flat_size])

            self.l4, self.pred_w['l4_w'], self.pred_w['l4_b'] = linear(self.l3, output_dim=1024)
            self.q, self.pred_w['pred_q_w'], self.pred_w['pred_q_b'] = linear(self.l4, output_dim=self.num_actions)

        self.target_w = {}
        with tf.variable_scope('target') :

            self.target_obs = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], 1], name='obs')
            self.target_l1, self.target_w['l1_w'], self.target_w['l1_b'] = conv2d(self.obs, output_dim=32, kernel_size=[8, 8], stride=[4, 4], name='l1')
            self.target_l2, self.target_w['l2_w'], self.target_w['l2_b'] = conv2d(self.target_l1, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l2')
            self.target_l3, self.target_w['l3_w'], self.target_w['l3_b'] = conv2d(self.target_l2, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l3')

            flat_size = reduce(lambda x,y : x*y, tf.get_shape(self.target_l3).as_list()[1:])
            self.l3_flat = tf.reshape(self.target_l3, [-1, flat_size])

            self.target_l4, self.target_w['l4_w'], self.target_w['l4_b'] = linear(self.target_l3, output_dim=1024)
            self.target_q, self.target_w['pred_q_w'], self.target_w['pred_q_b'] = linear(self.target_l4, output_dim=self.num_actions)

        with tf.variable_scope('pred_to_target') :
            self.pred2target_input = {}
            self.pred2target_assign_op = {}

            for name, w in self.target_w.keys() :
                self.pred2target_input[name] = tf.placeholder(tf.float32, shape=w.get_shape().as_list(), name=name)
                self.pred2target_assign_op[name] = w.assign(self.pred2target_input[name]))

        with tf.variable_scope('optimizer') :
            self.td_target = tf.placeholder(tf.float32, [None], name='target_q')
            self.action_ix = tf.placeholder(tf.float32, [None], name='action_ix')

            batch_size = tf.shape(self.q).as_list()[-1]
            gather_ix = tf.range(batch_size) * self.num_actions + self.action_ix
            self.chosen_actions = tf.gather(self.q, gather_ix)

            self.loss = tf.reduce_mean(tf.squared_difference(self.td_target, self.chosen_actions))
            optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
        with tf.variable_scope('summary') : 
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def predict(self, obs) :
        pred_Q = self.q.eval({self.obs : obs})
        return pred_Q

    def predict_target(self, obs) :
        pred_Q = self.target_q.eval({self.target_obs : obs})
        return pred_Q

    def update(self, obs, td_target, action_ix) :
        loss, _ = self.sess.run([self.loss, self.train_op], {self.obs:obs, self.td_target: td_target, self.action_ix: action_ix})
        return loss

    def update_target_network(self) :
        for w in self.w.keys() :
            sess.run(self.pred2target_assign_op[w], {self.pred2target_input[w] : self.w[w].eval()})


class ObsProcessor() :

    def __init__(self) :
        self.raw_imgs = tf.placeholder(tf.int8, shape=[210, 160, 3], name="raw_imgs")
        grayscale = tf.image.rgb_to_grayscale(self.raw_imgs)
        resized = tf.image.resize_images(
                        grayscale, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.output = tf.cast(resized, tf.float32)

    def preprocess(self, sess, obs, create_summary=False) :

        preprocessed = sess.run(self.output, {self.raw_imgs: obs})

        if create_summary :
            s = StringIO()
            plt.imsave(s, np.squeeze(preprocessed, axis=2) , format='png')
            img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                           height=preprocessed.shape[0],
                                           width=preprocessed.shape[1])
            return preprocessed, img_summary

        return preprocessed

def init_greedy_pi(Qnet, num_actions, ep) :
    def pi(sess, obs) :

        Qs = Qnet.predict(sess, obs)
        greedy_action = np.argmax(Qs)

        dist = np.ones(num_actions) * ep/num_actions
        dist[greedy_action] += 1 - ep

        return np.random.choice(num_actions, p=dist)
    return pi

def run_Q_learning(args) :

    env = gym.make('Breakout-v0')

    num_actions = env.action_space.n

    processor = ObsProcessor()
    net = Network(num_actions, [84, 84])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # populate initial replay memory

    replay_memory = []
    while len(replay_memory) < args.init_replay_size :
        done = False
        obs_t = env.reset()
        obs_t, img_sum = processor.preprocess(sess, obs_t)
        while not done :
            A_t = pi(sess, obs_t)
            obs_tp1, R_tp1, done, info = env.step(A_t)
            replay_memory.append([obs_t, A_t, obs_tp1, R_tp1, float(done)])
            obs_t, img_sum = processor.preprocess(sess, obs_tp1)

    iter_ix = 0 
    for ix in range(args.num_episodes) :

        done = False
        obs_t = env.reset()
        obs_t = processor.preprocess(sess, obs_t)
        while not done :
            iter_ix = iter_ix + 1

            if iter_ix % args.update_target_every == 0 :
                net.update_target_network()

            A_t = pi(sess, obs_t)
            obs_tp1, R_tp1, done, info = env.step(A_t)
            replay_memory.append([obs_t, A_t, obs_tp1, R_tp1, float(done)])

            # sample minibatch from replay_memory
            replay_sample = random.sample(replay_memory, args.batch_size)
            obs_batch, A_batch, obs_tp1_batch, R_batch, done_batch = map(np.array, zip(*replay_sample))

            # compute TD targets using target network 
            target_Qs = net.predict_target(sess, obs_tp1_batch)
            max_Qs = np.amax(target_Qs, axis=1)
            td_targets = reward_batch + (1. - done_batch) * discount_factor * max_Qs

            loss = net.update(obs_batch, td_targets, A_batch)

            obs_t = processor.preprocess(sess, obs_tp1)

            print('{} iter : {}'.format(iter_ix, loss))

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='arguments for dqn')

    parser.add_argument('-batch_size', default=4, dest='batch_size', type=int)
    parser.add_argument('-init_replay_size', default=64, dest='init_replay_size', type=int)
    parser.add_argument('-ep', default=.1, dest='ep', type=float)
    parser.add_argument('-summary_dir', default="./summary/", dest='summary_dir', type=str)
    parser.add_argument('-num_episodes', default=16, dest='num_episodes', type=int)
    parser.add_argument('-update_target_every', default=1000, dest='update_target_every', type=int)
    parser.add_argument('-discount_factor', default=.99, dest='discount_factor', type=float)

    args = parser.parse_args()

    run_Q_learning(args)