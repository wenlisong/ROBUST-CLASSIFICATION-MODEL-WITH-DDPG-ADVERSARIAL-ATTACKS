import os
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from collections import deque
import random
from tensorflow.contrib.slim.nets import inception
import time
from mytools import load_path_label
# import pdb


np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000

OUTPUT_GRAPH = True

tf.flags.DEFINE_string(
    'checkpoint_path', './defense_example/models/inception_v1/inception_v1.ckpt', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'ddpg_checkpoint_path', './models/ddpg/model', 'Path to checkpoint for ddpg network.')
tf.flags.DEFINE_string(
    'input_dir', './datasets/train_labels.txt', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_file', './output-defense.txt', 'Output file to save labels.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 32, 'Batch size to processing images')
tf.flags.DEFINE_integer(
    'nb_pixel', 224*224*3, 'The number of pixelx in a image')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'How many classes of the data set')
tf.app.flags.DEFINE_integer(
    'max_ep_steps', 100, 'The number of epoch times')
tf.app.flags.DEFINE_integer(
    'max_steps', 100000, 'The number of training times')
FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

config = tf.ConfigProto()
# allocate 50% of GPU memory
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
###############################  Actor  ####################################
class Actor(object):
    def __init__(self, sess, action_dim, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        # self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.epsilon = 1e-12

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            # net = tf.layers.dense(s, 30, activation=tf.nn.relu,
            #                       kernel_initializer=init_w, bias_initializer=init_b, name='l1',
            #                       trainable=trainable)
            # encoder
            conv1 = tf.layers.conv2d(s, 64, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='conv1', trainable=trainable)  # 224x224x64
            maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same', name='maxpool1')  # 112x112x64
            conv2 = tf.layers.conv2d(maxpool1, 32, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='conv2', trainable=trainable)  # 112x112x32
            maxpool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), 'same', name='maxpool2')  # 56x56x32
            conv3 = tf.layers.conv2d(maxpool2, 16, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='conv3', trainable=trainable)  #56x56x16
            maxpool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), 'same', name='maxpool3')  #28x28x16
            
            # decoder
            upsample1 = tf.image.resize_nearest_neighbor(maxpool3, (56, 56))
            deconv1 = tf.layers.conv2d_transpose(upsample1, 32, (3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='deconv1', trainable=trainable)
            upsample2 = tf.image.resize_nearest_neighbor(deconv1, (112, 112))
            deconv2 = tf.layers.conv2d_transpose(upsample2, 64, (3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='deconv2', trainable=trainable)
            upsample3 = tf.image.resize_nearest_neighbor(deconv2, (224, 224))
            deconv3 = tf.layers.conv2d_transpose(upsample3, 64, (3,3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='deconv3', trainable=trainable)

            with tf.variable_scope('a'):
                # actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                #                           bias_initializer=init_b, name='a', trainable=trainable)
                actions = tf.layers.conv2d(deconv3, 3, (3,3), padding='same', kernel_initializer=init_w, bias_initializer=init_b, name='a', trainable=trainable)
                # scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound

        return actions

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        action = self.sess.run(self.a, feed_dict={S: s})[0] # single action
        # scale action to [-1, 1]
        for i in range(action.shape[2]):
            cur_channel = action[:, :, i]
            min_val = np.min(cur_channel)
            max_val = np.max(cur_channel)
            action[:, :, i] = (cur_channel - min_val + self.epsilon) / max((max_val - min_val), 2 * self.epsilon)
        action = action * 2 -1
        return action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

###############################  Critic  ####################################
class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            def conv_desc(scope, input):
                with tf.variable_scope(scope):
                    conv1 = tf.layers.conv2d(input, 64, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='conv1', trainable=trainable)  # 224x224x64
                    maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same', name='maxpool1')  # 112x112x64
                    conv2 = tf.layers.conv2d(maxpool1, 32, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='conv2', trainable=trainable)  # 112x112x32
                    maxpool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), 'same', name='maxpool2')  # 56x56x32
                    conv3 = tf.layers.conv2d(maxpool2, 16, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='conv3', trainable=trainable)  #56x56x16
                    maxpool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), 'same', name='maxpool3')  #28x28x16
                    flatten = tf.layers.flatten(maxpool3)
                return flatten
            
            s = conv_desc('state_conv', s)
            a = conv_desc('action_conv', a)
            with tf.variable_scope('fcl1'):
                n_fcl1 = 1024
                w1_s = tf.get_variable('w1_s', [28*28*16, n_fcl1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [28*28*16, n_fcl1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_fcl1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

#####################  Memory  ####################
class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = deque()

    def store_transition(self, s, a, r):
        self.data.append((s, a, r))
        if len(self.data) > self.capacity:
            self.data.popleft()

    def sample(self, n):
        # assert len(self.data)>= self.capacity, 'Memory has not been fulfilled'
        # indices = np.random.choice(self.capacity, size=n)
        # return self.data[indices, :]
        return random.sample(self.data, n)

#####################  Load Image  ####################
def load_images(input_dir):
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
        with open(filepath, 'rb') as f:
            raw_image = imread(f, mode='RGB')
            image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
            image = (image / 255.0) * 2.0 - 1.0

        filename = os.path.basename(filepath)
        yield filename, image

#####################  Compute Reward  ####################
class Classifier(object):
    def __init__(self, input_shape, nb_classes):
        # self.graph = tf.Graph()
        self.sess = None
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.restore_model()

    def restore_model(self):
        with tf.Graph().as_default():
            # Prepare graph
            self.x_input = tf.placeholder(tf.float32, shape=self.input_shape)

            with tf.contrib.slim.arg_scope(inception.inception_v1_arg_scope()):
                _, end_points = inception.inception_v1(self.x_input, num_classes=self.nb_classes, is_training=False)
                self.pre_labels = tf.argmax(end_points['Predictions'], 1)
            # Restore Model
            saver = tf.train.Saver(tf.contrib.slim.get_model_variables())
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(saver=saver),
                checkpoint_filename_with_path=FLAGS.checkpoint_path)
            
            self.sess = tf.train.MonitoredSession(session_creator=session_creator)

    def get_reward(self, s, a, label):
        pre_labels = self.sess.run(self.pre_labels, feed_dict={self.x_input: a[tf.newaxis, :]})
        
        # import matplotlib.pyplot as plt
        # f = plt.figure()
        # f.add_subplot(1, 2, 1)
        # plt.imshow((s + 1.0) / 2.0)
        # f.add_subplot(1, 2, 2)
        # plt.imshow((a + 1.0) / 2.0)
        # plt.show(block=True)

        if pre_labels[0] == label:
            r = -1
        else:
            l2_dist = np.linalg.norm((a - s + 1.0) * 255.0 / 2.0)
            r = -np.power(2, l2_dist/128)+2
        return r

#####################  Main  ####################
if __name__ == "__main__":
    state_dim = [FLAGS.image_height, FLAGS.image_width, 3]
    action_dim = [FLAGS.image_height, FLAGS.image_width, 3]
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    # action_bound = None

    # all placeholder for tf
    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None]+state_dim, name='s')
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, None, name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None]+state_dim, name='s_')

    sess = tf.Session(config=config)
    # Create actor and critic
    actor = Actor(sess, action_dim, LR_A, REPLACEMENT)
    critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    sess.run(tf.global_variables_initializer())
    ac_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic')
    ac_saver = tf.train.Saver(ac_var_list, max_to_keep=3, keep_checkpoint_every_n_hours=1)

    # initialization classifier
    classifier = Classifier([None] + action_dim, FLAGS.num_classes)
    
    M = Memory(MEMORY_CAPACITY)

    # if OUTPUT_GRAPH:
    #     tf.summary.FileWriter("./logs/", sess.graph)

    var = 3  # control exploration

    start = time.time()
    for episode in range(FLAGS.max_ep_steps):
        ep_reward = 0
        data_generator = load_path_label(FLAGS.input_dir, [1, FLAGS.image_height, FLAGS.image_width, 3])
        for step in range(FLAGS.max_steps):
            (images, labels) = next(data_generator)
            # Add exploration noise
            a = actor.choose_action(images[0])
            a = np.clip(np.random.normal(a, var), -1, 1)    # add randomness to action selection for exploration
            # s_, r, done, info = env.step(a)
            r = classifier.get_reward(images[0], a, labels[0])

            M.store_transition(images[0], a, r / 10)

            if len(M.data) > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                minibatch = M.sample(FLAGS.batch_size)
                b_s = [row[0] for row in minibatch]
                b_a = [row[1] for row in minibatch]
                b_r = [row[2] for row in minibatch]
                b_s_ = b_a

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            # s = s_
            ep_reward += r

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size) /(time.time() - start)
                start = time.time()
                print('Episode:{}, Step {:06d}, {:.2f} seconds/step, {:.2f} examples/second, ep_reward: {:.3f}, Explore: {:.3f}'.format(
                    episode, step, avg_time_per_step,
                    avg_examples_per_second, ep_reward, var))

        ac_saver.save(sess, FLAGS.ddpg_checkpoint_path, global_step=episode)
        print('Running time: ', time.time() - start)