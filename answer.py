def optimize_iou(nn_last_layer, correct_label, learning_rate, num_classes):

"""
Build the TensorFLow loss and optimizer operations.
:param nn_last_layer: TF Tensor of the last layer in the neural network
:param correct_label: TF Placeholder for the correct label image
:param learning_rate: TF Placeholder for the learning rate
:param num_classes: Number of classes to classify
:return: Tuple of (logits, train_op, cross_entropy_loss)
"""

logits = tf.reshape(nn_last_layer, (-1, num_classes))    
labels = tf.reshape(correct_label, (-1, num_classes))    

# Cross entropy
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))

#cross_entropy_loss = tf.reduce_mean(iou)

# Training loss
loss = tf.reduce_mean(cross_entropy_loss)
tf.add_to_collection('all_iou',tf.multiply(1/25,loss))

iou,iouop = tf.metrics.mean_iou(labels,logits, num_classes,name = 'iou')
with tf.control_dependencies([iouop]):
    mod_iou = tf.subtract(tf.constant(1.0),iou)
    tf.add_to_collection('all_iou',mod_iou)

mod_loss = tf.reduce_sum(tf.stack(tf.get_collection('all_iou')))
print("loss:",loss)
print("mod_loss:",mod_loss)

global_step = tf.Variable(0, name='global_step', trainable=False) #learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op =  optimizer.minimize(mod_loss, var_list=get_trainable_variables(), global_step=global_step)

return logits, train_op, cross_entropy_loss
