import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
from mytools import load_path_label
from tqdm import tqdm

tf.flags.DEFINE_string(
    'checkpoint_path', './defense_example/models/inception_v1/inception_v1.ckpt', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'test_data_dir', './datasets/test_labels.txt', 'Input directory with images.')
tf.flags.DEFINE_string(
    'train_data_dir', './datasets/train_labels.txt', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_file', None, 'Output file to save labels.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'Batch size to processing images')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'How many classes of the data set')
FLAGS = tf.flags.FLAGS

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def main(argv=None):
    print("This script is used to compute accuracy!")
    if len(argv) < 2:
        print("No argv! You need to assign train/test data as below:")
        print("Try: python compute_acc.py train, python compute_acc.py test\n")
        return

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    nb_classes = FLAGS.num_classes

    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    # allocate 50% of GPU memory
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Graph().as_default():
        print("Prepare graph...")
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        noise_input = gaussian_noise_layer(x_input, .2)

        with slim.arg_scope(inception.inception_v1_arg_scope()):
            _, end_points = inception.inception_v1(
                noise_input, num_classes=nb_classes, is_training=False)
        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        print("Restore Model...")
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            config=config,
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path)
        
        print("Run computation...")
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            if argv[1] == 'test':
                INPUT_DIR = FLAGS.test_data_dir
                FLAGS.output_file = './result/test_accuracy.txt'
            elif argv[1] == 'train':
                INPUT_DIR = FLAGS.train_data_dir
                FLAGS.output_file = './result/train_accuracy.txt'

            data_generator = load_path_label(INPUT_DIR, batch_shape, shuffle=False)

            acc_dict = {}
            for i in range(FLAGS.num_classes):
                acc_dict[i] = [0, 0]
            for images, true_labels in tqdm(data_generator):
                labels = sess.run(predicted_labels, feed_dict={x_input: images})
                for i in range(len(true_labels)):
                    acc_dict[true_labels[i]][1] += 1
                    if labels[i] == true_labels[i]:
                        acc_dict[true_labels[i]][0] += 1
            
            print("Compute accuracy...")
            with open(FLAGS.output_file, 'w') as f:
                total_true = 0
                total_count = 0
                for i in range(FLAGS.num_classes):
                    total_true += acc_dict[i][0]
                    total_count += acc_dict[i][1]
                    f.writelines("class: %d, accuracy: %d/%d = %.3f \n" % (i, acc_dict[i][0], acc_dict[i][1], acc_dict[i][0] / acc_dict[i][1]))
                
                print("Total accuracy: %.3f \n" % (total_true / total_count))
                f.writelines("Total accuracy: %.3f \n" % (total_true / total_count))
            
            print('Save accuracy result to %s' %FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
