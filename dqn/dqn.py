import gym
import tensorflow as tf
import numpy as np 
import random
import argparse

from StringIO import StringIO
import matplotlib.pyplot as plt 

from utils import net 

tf.set_random_seed(123)

class DQN() :

    def __init__(self, sess, num_actions, img_size, summary_dir=None) :
        self.sess = sess 
        self.num_actions = num_actions 
        self.img_size = img_size

        self.build_dqn()

        self.fw = None
        if summary_dir :
            self.fw = tf.summary.FileWriter(summary_dir)

    def build_dqn(self) :

        initializer = tf.truncated_normal_initializer(0, 0.02)
        global_step = tf.Variable(0, name='global_step')

        self.pred_w = {}
        with tf.variable_scope('prediction') :

            self.obs = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], 4], name='obs')

            self.l1, self.pred_w['l1_w'], self.pred_w['l1_b'] = net.conv2d(self.obs, output_dim=32, kernel_size=[8, 8], stride=[4, 4], name='l1')
            self.l2, self.pred_w['l2_w'], self.pred_w['l2_b'] = net.conv2d(self.l1, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l2')
            self.l3, self.pred_w['l3_w'], self.pred_w['l3_b'] = net.conv2d(self.l2, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l3')

            flat_size = reduce(lambda x,y : x*y, self.l3.get_shape().as_list()[1:])
            self.l3_flat = tf.reshape(self.l3, [-1, flat_size])

            self.l4, self.pred_w['l4_w'], self.pred_w['l4_b'] = net.linear(self.l3_flat, output_dim=1024, name='l4')
            self.q, self.pred_w['pred_q_w'], self.pred_w['pred_q_b'] = net.linear(self.l4, output_dim=self.num_actions, name='pred')

        self.target_w = {}
        with tf.variable_scope('target') :

            self.target_obs = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], 4], name='obs')
            self.target_l1, self.target_w['l1_w'], self.target_w['l1_b'] = net.conv2d(self.target_obs, output_dim=32, kernel_size=[8, 8], stride=[4, 4], name='l1')
            self.target_l2, self.target_w['l2_w'], self.target_w['l2_b'] = net.conv2d(self.target_l1, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l2')
            self.target_l3, self.target_w['l3_w'], self.target_w['l3_b'] = net.conv2d(self.target_l2, output_dim=64, kernel_size=[4, 4], stride=[2, 2], name='l3')

            flat_size = reduce(lambda x,y : x*y, self.target_l3.get_shape().as_list()[1:])
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, flat_size])

            self.target_l4, self.target_w['l4_w'], self.target_w['l4_b'] = net.linear(self.target_l3_flat, output_dim=1024, name='l4')
            self.target_q, self.target_w['pred_q_w'], self.target_w['pred_q_b'] = net.linear(self.target_l4, output_dim=self.num_actions, name='pred')

        with tf.variable_scope('pred_to_target') :
            self.pred2target_input = {}
            self.pred2target_assign_op = {}

            for name, target_w in self.target_w.iteritems() :
                self.pred2target_input[name] = tf.placeholder(tf.float32, shape=target_w.get_shape().as_list(), name=name)
                self.pred2target_assign_op[name] = tf.assign(target_w, self.pred2target_input[name])

        with tf.variable_scope('optimizer') :
            self.td_target = tf.placeholder(tf.float32, [None], name='target_q')
            self.action_ix = tf.placeholder(tf.int32, [None], name='action_ix')

            batch_size = tf.shape(self.q)[0]
            gather_ix = tf.range(batch_size) * self.num_actions + self.action_ix
            self.chosen_actions_vals = tf.gather(tf.reshape(self.q, [-1]), gather_ix)

            self.loss = tf.reduce_mean(net.clipped_error(self.td_target - self.chosen_actions_vals))
            optimizer = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01 )
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        scalar_summary_tags = ['loss', 'avg_loss']

        with tf.variable_scope('summary') :
            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags :
                self.summary_placeholders[tag] = tf.placeholder(tf.float32, None)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

    def predict(self, obs) :
        net_input = obs 
        if obs.ndim == 3 :
            net_input = np.expand_dims(obs, axis=0)

        pred_Q = self.sess.run(self.q, {self.obs: net_input}) 
        return pred_Q

    def predict_target(self, obs) :
        pred_Q = self.sess.run(self.target_q, {self.target_obs: obs})
        return pred_Q

    def update(self, obs, td_target, action_ix) :
        loss, _ = self.sess.run([self.loss, self.train_op], {self.obs:obs, self.td_target: td_target, self.action_ix: action_ix})
        return loss

    def update_target_network(self) :
        for w in self.pred_w.keys() :
            self.sess.run(self.pred2target_assign_op[w], {self.pred2target_input[w] : self.sess.run(self.pred_w[w])})

    def inject_summary(self, summary_dict, step) :

        summary_protobuf = self.sess.run([self.summary_ops[tag] for tag in summary_dict.keys()], 
                                {self.summary_placeholders[tag]:val for tag,val in summary_dict.iteritems()})
        for summary in summary_protobuf :
            self.fw.add_summary(summary, step)


class ObsProcessor() :

    def __init__(self, sess) :
        self.sess = sess 
        self.raw_imgs = tf.placeholder(tf.int8, shape=[210, 160, 3], name="raw_imgs")
        grayscale = tf.image.rgb_to_grayscale(self.raw_imgs)
        resized = tf.image.resize_images(
                        grayscale, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.output = tf.cast(tf.squeeze(resized, axis=2), tf.float32)

    def preprocess(self, obs, create_summary=False) :

        preprocessed = self.sess.run(self.output, {self.raw_imgs: obs})

        if create_summary :
            s = StringIO()
            plt.imsave(s, np.squeeze(preprocessed, axis=2), format='png')
            img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                           height=preprocessed.shape[0],
                                           width=preprocessed.shape[1])
            return preprocessed, img_summary

        return preprocessed

def init_greedy_pi(Qnet, num_actions) :
    def pi(obs, ep) :

        Qs = Qnet.predict(obs)
        greedy_action = np.argmax(Qs)

        dist = np.ones(num_actions) * ep/num_actions
        dist[greedy_action] += 1 - ep

        return np.random.choice(num_actions, p=dist)
    return pi

def run_Q_learning(args) :

    env = gym.make('Breakout-v0')
    num_actions = env.action_space.n

    sess = tf.Session()

    processor = ObsProcessor(sess)
    dqn = DQN(sess, num_actions, [args.state_size, args.state_size], args.summary_dir)
    pi = init_greedy_pi(dqn, num_actions)
    epsilons = np.linspace(1.0, args.ep_end, args.ep_decay_steps)

    # populate initial replay memory

    # replay_memory = []
    # while len(replay_memory) < args.init_replay_size :
    #     # start of episode initialization
    #     s_t = np.stack([np.zeros(args.state_size, args.state_size)] * 4, axis=2) 
    #     done = False

    #     # start episode
    #     obs_t = env.reset()
    #     obs_t = processor.preprocess(obs_t) 
    #     s_t = np.append(s_t[:,:,1:], np.expand_dims(s_t, 2), axis=2)
    #     while not done :
    #         A_t = pi(s_t, 1.0)
    #         obs_tp1, R_tp1, done, info = env.step(A_t)
    #         obs_tp1 = processor.preprocess(obs_tp1)
    #         s_tp1 = np.append(s_t[:,:,1:], np.expand_dims(obs_tp1, 2), axis=2)
    #         replay_memory.append([s_t, A_t, s_tp1, R_tp1, float(done)])

    #         s_t = s_tp1

    replay_memory = []
    while len(replay_memory) < args.init_replay_size :
        s_t = np.stack([np.zeros(args.state_size, args.state_size)] * 4, axis=2)
        done = False 

        obs_t = env.reset()
        obs_t = processor.preprocess(obs_t)
        frame_count = 1 
        while True :
            

    iter_ix = 0
    total_loss = 0.
    for ix in range(args.num_episodes) :

        done = False
        obs_t = env.reset()
        obs_t = processor.preprocess(obs_t)
        s_t = np.stack([obs_t] * 4, axis=2)
        while not done :
            iter_ix = iter_ix + 1

            if iter_ix % args.update_target_every == 0 :
                dqn.update_target_network()

            ep = args.ep_end if iter_ix > len(epsilons) else epsilons[iter_ix]

            A_t = pi(s_t, ep)
            obs_tp1, R_tp1, done, info = env.step(A_t)
            obs_tp1 = processor.preprocess(obs_tp1)
            s_tp1 = np.append(s_t[:,:,1:], np.expand_dims(obs_tp1, 2), axis=2)
            replay_memory.append([s_t, A_t, s_tp1, R_tp1, float(done)])

            # sample minibatch from replay_memory
            replay_sample = random.sample(replay_memory, args.batch_size)
            obs_batch, A_batch, obs_tp1_batch, R_batch, done_batch = map(np.array, zip(*replay_sample))

            # compute TD targets using target network 
            target_Qs = dqn.predict_target(obs_tp1_batch)
            max_Qs = np.max(target_Qs, axis=1)
            td_targets = R_batch + (1. - done_batch) * args.discount_factor * max_Qs

            loss = dqn.update(obs_batch, td_targets, A_batch)
            total_loss += loss 

            dqn.inject_summary({'loss': loss, 'avg_loss': total_loss/iter_ix}, iter_ix)

            s_t = s_tp1

            if iter_ix % args.print_every == 0 : 
                print('{} iter : {}'.format(iter_ix, loss))

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='arguments for dqn')

    parser.add_argument('-batch_size', default=4, dest='batch_size', type=int)
    parser.add_argument('-num_episodes', default=16, dest='num_episodes', type=int)
    parser.add_argument('-discount_factor', default=.99, dest='discount_factor', type=float)
    parser.add_argument('-init_replay_size', default=64, dest='init_replay_size', type=int)
    parser.add_argument('-update_target_every', default=1000, dest='update_target_every', type=int)
    parser.add_argument('-ep_end', default=.1, dest='ep_end', type=float)
    parser.add_argument('-ep_decay_steps', default=500000, dest='ep_decay_steps', type=int)
    parser.add_argument('-frame_skip', default=4, dest='frame_skip', type=int)
    parser.add_argument('-state_size', default=84, dest='init_replay_size', type=int)

    parser.add_argument('-summary_dir', default="./summary/", dest='summary_dir', type=str)
    parser.add_argument('-print_every', default=1000, dest='print_every', type=int)

    args = parser.parse_args()

    run_Q_learning(args)
