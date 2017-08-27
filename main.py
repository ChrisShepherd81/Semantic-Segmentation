import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

######################################################################################################################
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
        
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    
    #shape(1, 20, 72, 256)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    
    #shape(1, 10, 36, 512)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    
    #shape(1, 5, 18, 4096)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out  
#tests.test_load_vgg(load_vgg, tf)
######################################################################################################################

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    initializer = tf.truncated_normal_initializer(stddev = 0.01) # 0.001, _0.01_
    ##########################################################################################################
    with tf.name_scope("conf_decoder_layer1"):
        #shape(1, 5, 18, 2)
        conf_decoder_layer1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), 
                                               kernel_initializer=initializer)
    ##########################################################################################################
    with tf.name_scope("conf_decoder_layer2"):
        #shape(1, 10, 36, 2)
        conf_decoder_layer2_up = tf.layers.conv2d_transpose(conf_decoder_layer1, num_classes, 2, strides=(2, 2), 
                                                            padding='SAME', kernel_initializer=initializer)
        #Add skip layer 4_out
        conf_decoder_layer2_skip = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), 
                                                   kernel_initializer=initializer)
        conf_decoder_layer2 = tf.add(conf_decoder_layer2_up, conf_decoder_layer2_skip)
    
    ##########################################################################################################
    with tf.name_scope("conf_decoder_layer3"):
        #shape(1, 20, 72, 2)
        conf_decoder_layer3_up = tf.layers.conv2d_transpose(conf_decoder_layer2, num_classes, 2, strides=(2, 2), 
                                                            padding='SAME', kernel_initializer=initializer)
        #Add skip layer 3_out
        conf_decoder_layer3_skip = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), 
                                                    kernel_initializer=initializer)
        conf_decoder_layer3 = tf.add(conf_decoder_layer3_up, conf_decoder_layer3_skip)
    ##########################################################################################################
    with tf.name_scope("conf_decoder_layer4"):
        #shape(1, 40, 144, 2)
        conf_decoder_layer4 = tf.layers.conv2d_transpose(conf_decoder_layer3, num_classes, 2, strides=(2, 2), 
                                            padding='SAME', kernel_initializer=initializer)
    ##########################################################################################################
    with tf.name_scope("conf_decoder_layer5"):
        #shape(1, 80, 288, 2)
        conf_decoder_layer5 = tf.layers.conv2d_transpose(conf_decoder_layer4, num_classes, 2, strides=(2, 2), 
                                            padding='SAME', kernel_initializer=initializer)
    ##########################################################################################################
    with tf.name_scope("conf_output_layer"):
        #shape(1, 160, 576, 2)
        output_layer = tf.layers.conv2d_transpose(conf_decoder_layer5, num_classes, 2, strides=(2, 2), 
                                            padding='SAME', kernel_initializer=initializer)
    ##########################################################################################################
    return output_layer
#tests.test_layers(layers)
######################################################################################################################

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
   
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    
    return logits, train_op, cross_entropy_loss
#tests.test_optimize(optimize)
######################################################################################################################

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    kp = 0.7 #keep_prob: 0.5, 0.6, _0.7_, 0.9
    lr = 0.001 #learning_rate: 0.0001, _0.001_, 0.01
    
    train_writer = tf.summary.FileWriter('./log/' + str(time.time()), sess.graph)
    tf.summary.scalar("batch_size", batch_size)
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("keep_prob", keep_prob)
    tf.summary.scalar("epochs", epochs)
       
    overall_cnt = 0
    for i in range(epochs):
        print("Epoch " + str(i+1) + " running ...")
        epoch_loss = 0;
        batch_cnt = 0;
        for batch_img, batch_label in (get_batches_fn(batch_size)):             
            merge_op = tf.summary.merge_all()
            summary, _, loss = sess.run([merge_op, train_op, cross_entropy_loss], feed_dict={input_image: batch_img, correct_label: batch_label, keep_prob: kp, learning_rate: lr})
            train_writer.add_summary(summary, overall_cnt)
            epoch_loss += loss
            overall_cnt += 1
            batch_cnt += 1
        
        print("Epoch Loss: " + str(epoch_loss/batch_cnt))
        
    train_writer.close()
#tests.test_train_nn(train_nn)
######################################################################################################################

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)
    epochs = 10 # _10_, 20, 100
    #Maximum of 10. No bigger batch size possible on GTX 1060
    batch_size = 8 # 2, 4, 6, _8_, 10

    # Download pre-trained VGG model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    # Path to VGG model
    vgg_path = os.path.join(data_dir, 'vgg')

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)  
          
    correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], num_classes))
    learning_rate = tf.placeholder(tf.float32)
    
    with tf.Session() as sess:
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes) 
        
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
    
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
    
        #Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
######################################################################################################################

if __name__ == '__main__':
    run()
    print("Exit program")
