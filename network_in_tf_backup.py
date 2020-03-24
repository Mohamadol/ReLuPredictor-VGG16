import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def tf_model(x):
    
    conv1 = tf.nn.conv2d(x, w1[0], strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, w1[1])
    rel1 = tf.nn.relu(conv1)
    batch_norm1 =tf.layers.batch_normalization(rel1,beta_initializer=tf.constant_initializer(bn1[1]),gamma_initializer=tf.constant_initializer(bn1[0]),moving_mean_initializer=tf.constant_initializer(bn1[2]),moving_variance_initializer=tf.constant_initializer(bn1[3]))
    conv2 = tf.nn.conv2d(batch_norm1, w2[0], strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, w2[1])
    rel2 = tf.nn.relu(conv2)
    batch_norm2 =tf.layers.batch_normalization(rel2,beta_initializer=tf.constant_initializer(bn2[1]),gamma_initializer=tf.constant_initializer(bn2[0]),moving_mean_initializer=tf.constant_initializer(bn2[2]),moving_variance_initializer=tf.constant_initializer(bn2[3]))
    pool1 = tf.nn.max_pool(batch_norm2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID');
    
    
    conv3 = tf.nn.conv2d(pool1, w3[0], strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, w3[1])
    rel3 = tf.nn.relu(conv3)
    batch_norm3 =tf.layers.batch_normalization(rel3,beta_initializer=tf.constant_initializer(bn3[1]),gamma_initializer=tf.constant_initializer(bn3[0]),moving_mean_initializer=tf.constant_initializer(bn3[2]),moving_variance_initializer=tf.constant_initializer(bn3[3]))
    conv4 = tf.nn.conv2d(batch_norm3, w4[0], strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, w4[1])
    rel4 = tf.nn.relu(conv4)
    batch_norm4 =tf.layers.batch_normalization(rel4,beta_initializer=tf.constant_initializer(bn4[1]),gamma_initializer=tf.constant_initializer(bn4[0]),moving_mean_initializer=tf.constant_initializer(bn4[2]),moving_variance_initializer=tf.constant_initializer(bn4[3]))
    pool2 = tf.nn.max_pool(batch_norm4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID');
    
    
    conv5 = tf.nn.conv2d(pool2, w5[0], strides=[1,1,1,1], padding='SAME')
    conv5 = tf.nn.bias_add(conv5, w5[1])
    rel5 = tf.nn.relu(conv5)
    batch_norm5 =tf.layers.batch_normalization(rel5,beta_initializer=tf.constant_initializer(bn5[1]),gamma_initializer=tf.constant_initializer(bn5[0]),moving_mean_initializer=tf.constant_initializer(bn5[2]),moving_variance_initializer=tf.constant_initializer(bn5[3]))
    conv6 = tf.nn.conv2d(batch_norm5, w6[0], strides=[1,1,1,1], padding='SAME')
    conv6 = tf.nn.bias_add(conv6, w6[1])
    rel6 = tf.nn.relu(conv6)
    batch_norm6 =tf.layers.batch_normalization(rel6,beta_initializer=tf.constant_initializer(bn6[1]),gamma_initializer=tf.constant_initializer(bn6[0]),moving_mean_initializer=tf.constant_initializer(bn6[2]),moving_variance_initializer=tf.constant_initializer(bn6[3]))
    conv7 = tf.nn.conv2d(batch_norm6, w7[0], strides=[1,1,1,1], padding='SAME')
    conv7 = tf.nn.bias_add(conv7, w7[1])
    rel7 = tf.nn.relu(conv7)
    batch_norm7 =tf.layers.batch_normalization(rel7,beta_initializer=tf.constant_initializer(bn7[1]),gamma_initializer=tf.constant_initializer(bn7[0]),moving_mean_initializer=tf.constant_initializer(bn7[2]),moving_variance_initializer=tf.constant_initializer(bn7[3]))
    pool3 = tf.nn.max_pool(batch_norm7, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID');
    
    
    conv8 = tf.nn.conv2d(pool3, w8[0], strides=[1,1,1,1], padding='SAME')
    conv8 = tf.nn.bias_add(conv8, w8[1])
    rel8 = tf.nn.relu(conv8)
    batch_norm8 = tf.layers.batch_normalization(rel8,beta_initializer=tf.constant_initializer(bn8[1]),gamma_initializer=tf.constant_initializer(bn8[0]),moving_mean_initializer=tf.constant_initializer(bn8[2]),moving_variance_initializer=tf.constant_initializer(bn8[3]))
    conv9 = tf.nn.conv2d(batch_norm8, w9[0], strides=[1,1,1,1], padding='SAME')
    conv9 = tf.nn.bias_add(conv9, w9[1])
    rel9 = tf.nn.relu(conv9)
    batch_norm9 = tf.layers.batch_normalization(rel9,beta_initializer=tf.constant_initializer(bn9[1]),gamma_initializer=tf.constant_initializer(bn9[0]),moving_mean_initializer=tf.constant_initializer(bn9[2]),moving_variance_initializer=tf.constant_initializer(bn9[3]))
    conv10 = tf.nn.conv2d(batch_norm9, w10[0], strides=[1,1,1,1], padding='SAME')
    conv10 = tf.nn.bias_add(conv10, w10[1])
    rel10 = tf.nn.relu(conv10)
    batch_norm10 =tf.layers.batch_normalization(rel10,beta_initializer=tf.constant_initializer(bn10[1]),gamma_initializer=tf.constant_initializer(bn10[0]),moving_mean_initializer=tf.constant_initializer(bn10[2]),moving_variance_initializer=tf.constant_initializer(bn10[3]))
    pool4 = tf.nn.max_pool(batch_norm10, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID');
    
    
    conv11 = tf.nn.conv2d(pool4, w11[0], strides=[1,1,1,1], padding='SAME')
    conv11 = tf.nn.bias_add(conv11, w11[1])
    rel11 = tf.nn.relu(conv11)
    batch_norm11 =tf.layers.batch_normalization(rel11,beta_initializer=tf.constant_initializer(bn11[1]),gamma_initializer=tf.constant_initializer(bn11[0]),moving_mean_initializer=tf.constant_initializer(bn11[2]),moving_variance_initializer=tf.constant_initializer(bn11[3]))
    conv12 = tf.nn.conv2d(batch_norm11, w12[0], strides=[1,1,1,1], padding='SAME')
    conv12 = tf.nn.bias_add(conv12, w12[1])
    rel12 = tf.nn.relu(conv12)
    batch_norm12 =tf.layers.batch_normalization(rel12,beta_initializer=tf.constant_initializer(bn12[1]),gamma_initializer=tf.constant_initializer(bn12[0]),moving_mean_initializer=tf.constant_initializer(bn12[2]),moving_variance_initializer=tf.constant_initializer(bn12[3]))
    conv13 = tf.nn.conv2d(batch_norm12, w13[0], strides=[1,1,1,1], padding='SAME')
    conv13 = tf.nn.bias_add(conv13, w13[1])
    rel13 = tf.nn.relu(conv13)
    batch_norm13 = tf.layers.batch_normalization(rel13,beta_initializer=tf.constant_initializer(bn13[1]),gamma_initializer=tf.constant_initializer(bn13[0]),moving_mean_initializer=tf.constant_initializer(bn13[2]),moving_variance_initializer=tf.constant_initializer(bn13[3]))
    pool5 = tf.nn.max_pool(batch_norm13, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID');
    
    
    flat = tf.layers.flatten(pool5)  
    dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.constant_initializer(w14[0]),bias_initializer=tf.constant_initializer(w14[1]))
    #batch_norm14 =bn14[0]* (dense1 - bn14[2]) / np.sqrt(bn14[3] + 1e-3) + bn14[1]
    batch_norm14 = tf.layers.batch_normalization(dense1,beta_initializer=tf.constant_initializer(bn14[1]),gamma_initializer=tf.constant_initializer(bn14[0]),moving_mean_initializer=tf.constant_initializer(bn14[2]),moving_variance_initializer=tf.constant_initializer(bn14[3]))
    dense2 = tf.layers.dense(batch_norm14, 10, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(w15[0]),bias_initializer=tf.constant_initializer(w15[1]))
    tf.keras.activations.softmax(dense2)
    return dense2