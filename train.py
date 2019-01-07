"""
vehicles detection
X: image (640, 960, 3)
y: mask (640, 960, 1)
   - background is masked 0
   - vehicle is masked 255

Loss function: maximize IOU

"""
import time
import os
import pandas as pd
import tensorflow as tf
import sys

def image_augmentation(image, mask):

    concat_image = tf.concat([image, mask], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(concat_image)
    maybe_flipped = tf.image.random_flip_up_down(concat_image)

    image = maybe_flipped[:, :, :-1]
    mask = maybe_flipped[:, :, -1:]

    image = tf.image.random_brightness(image, 0.7)
    image = tf.image.random_hue(image, 0.3)

    return image, mask


def get_image_mask(queue, augmentation=True):

    text_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = text_reader.read(queue)

    image_path, mask_path = tf.decode_csv(
        csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    mask_file = tf.read_file(mask_path)

    image = tf.image.decode_jpeg(image_file, channels=3)
    image.set_shape([640, 960, 3])
    image = tf.cast(image, tf.float32)

    mask = tf.image.decode_jpeg(mask_file, channels=1)
    mask.set_shape([640, 960, 1])
    mask = tf.cast(mask, tf.float32)
    mask = mask / (tf.reduce_max(mask) + 1e-7)

    if augmentation:
        image, mask = image_augmentation(image, mask)

    return image, mask


def conv_conv_pool(input,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):

    net = input

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_concat(inputA, input_B, n_filter, flags, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, flags, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, flags, name):
    """Up Convolution `tensor` by 2 times
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
        name="upsample_{}".format(name))


def make_unet(X, training, flags=None):
    """U-Net architecture
    """
    net = X / 127.5 - 1.0
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, flags, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, flags, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, flags, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, flags, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, flags, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8, flags, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def make_train_op(y_pred, y_true):
    """Returns a training operation
    Returns:
        train_op: minimize operation
    """
    loss = -IOU_(y_pred, y_true)

    global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer()
    return optim.minimize(loss, global_step=global_step)


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--epochs", default=3, type=int, help="Number of epochs")

    parser.add_argument("--batch-size", default=4, type=int, help="Batch size")

    parser.add_argument(
        "--logdir", default="logs", help="Tensorboard log directory")

    parser.add_argument(
        "--reg", type=float, default=0.1, help="L2 Regularizer Term")

    parser.add_argument(
        "--ckdir", default="models", help="Checkpoint directory")

    flags = parser.parse_args()
    return flags


def main(flags):
    train = pd.read_csv("./train.csv")
    n_train = train.shape[0]

    test = pd.read_csv("./test.csv")
    n_test = test.shape[0]

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 640, 960, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 640, 960, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")

    pred = make_unet(X, mode, flags)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    # tf.summary.histogram("Predicted Mask", pred)
    tf.summary.image("input_image", X, max_outputs=2)
    tf.summary.image("Predict", pred, max_outputs=2)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, y)

    IOU_op = IOU_(pred, y)
    # IOU_op = tf.Print(IOU_op, [IOU_op])
    # tf.summary.scalar("IOU", IOU_op)

    train_csv = tf.train.string_input_producer(['train.csv'])
    test_csv = tf.train.string_input_producer(['test.csv'])
    train_image, train_mask = get_image_mask(train_csv)
    test_image, test_mask = get_image_mask(test_csv, augmentation=False)

    X_batch_op, y_batch_op = tf.train.shuffle_batch(
        [train_image, train_mask],
        batch_size=flags.batch_size,
        capacity=flags.batch_size * 10,
        min_after_dequeue=flags.batch_size * 5,
        allow_smaller_final_batch=True)

    X_test_op, y_test_op = tf.train.batch(
        [test_image, test_mask],
        batch_size=flags.batch_size,
        capacity=flags.batch_size * 10,
        allow_smaller_final_batch=True)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(flags.logdir, sess.graph)
        # test_summary_writer = tf.summary.FileWriter(test_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        try:
            global_step = tf.train.get_global_step(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(flags.epochs):
                start_time = time.time()
                train_iou=0.
                for step in range(0, n_train, flags.batch_size):

                    X_batch, y_batch = sess.run([X_batch_op, y_batch_op])

                    _, step_iou, global_step_value = sess.run(
                        [train_op, IOU_op, global_step],
                        feed_dict={X: X_batch,
                                   y: y_batch,
                                   mode: True})
                    # train_iou+=step_iou
                    print('------Epoch:%d,step:%d,train iou:%f'%(epoch, step, step_iou))
                    train_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_iou", simple_value=step_iou)
                    ])
                    summary_writer.add_summary(train_summary,global_step_value)
                    sys.stdout.flush()
                # print("Epoches %d took %fs" % (epoch, time.time() - start_time))
                # print("   train IOU: %f" % train_iou/n_train)
                total_iou = 0
                test_iou=0
                for step in range(0, n_test, flags.batch_size):
                    X_test, y_test = sess.run([X_test_op, y_test_op])
                    step_iou, step_summary = sess.run(
                        [IOU_op, summary_op],
                        feed_dict={X: X_test,
                                   y: y_test,
                                   mode: False})
                    print('------test iou:', step_iou)
                    # test_iou+=step_iou
                    # total_iou += step_iou * X_test.shape[0]
                    test_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="test_iou", simple_value=step_iou)
                    ])
                    summary_writer.add_summary(test_summary, (epoch + 1) * (step + 1))
                    summary_writer.add_summary(step_summary,(epoch + 1) * (step + 1))
                    sys.stdout.flush()
                # print("   test IOU: %f" % test_iou / n_test)
                saver.save(sess, "{}/model.ckpt".format(flags.logdir), epoch)
                # saver.save(sess, flags.ckdir + "model.ckpt", epoch)
                sys.stdout.flush()
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.logdir))


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
