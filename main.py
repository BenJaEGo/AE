from tensorflow.examples.tutorials.mnist import input_data
from auto_encoder import *
from vis_utils import *
import os


def run_training():
    save_path = 'samples'
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_input = n_output = 784
    n_encoder_units = [256]
    n_latent = 128
    n_decoder_units = [256]
    lam = 0.0001
    lr = 0.001

    max_epoch = 400
    batch_size = 256
    n_sample, n_dims = mnist.train.images.shape
    n_batch_each_epoch = n_sample // batch_size

    graph = tf.Graph()

    with graph.as_default():

        model = AutoEncoder(n_input, n_encoder_units, n_latent, n_decoder_units, n_output,
                            lr, lam,
                            tf.nn.tanh)

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(max_epoch):
                aver_loss = 0.0
                for step in range(n_batch_each_epoch):
                    x, y = mnist.train.next_batch(batch_size)
                    feed_dict = {model.x_pl: x}

                    tr_loss, _ = sess.run(
                        fetches=[model.loss, model.train_op],
                        feed_dict=feed_dict
                    )
                    aver_loss += tr_loss
                print("epoch %d, loss %f" % (epoch, aver_loss / n_batch_each_epoch))

                te_x, te_y = mnist.test.next_batch(16)
                samples = sess.run(fetches=[model.reconstruct],
                                   feed_dict={model.x_pl: te_x})

                fig = visualize_generate_samples(samples[0])
                plt.savefig('{path}/epoch_{epoch}.png'.format(
                    path=save_path, epoch=epoch), bbox_inches='tight')
                # plt.savefig('out/{}.png'.format(str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
