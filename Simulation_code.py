import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import math as m
import matplotlib.pyplot as plt
%matplotlib inline

#nfeatures is the inputs dimension.
nfeatures = 3

#Due to the fixed seed the created data is allways the same.
#If one time batch size is chosen as the number n and another time chosen as n+1
#then the first n data points are the same.
def create_data(batch_size, stddev):
    
    #Here the data is initialized:
    #Input:
    input_learn = tf.random_uniform([batch_size, nfeatures], minval=0, maxval=100, seed=3140, name="input")

    #Output:
    factors = tf.constant([[2], [-5], [4]], dtype="float32")
    norm = tf.random_normal([batch_size, 1], mean=0, stddev=stddev, seed=8235)
    output_learn3 = tf.linalg.matmul(input_learn, factors)
    output_learn2 = tf.math.add(output_learn3, tf.constant(30.0, shape=[batch_size, 1]))
    output_learn = tf.math.add(output_learn2, norm, name="output")

    
    #Here the data is created:
    sess = tf.InteractiveSession()
    input_learn_run = sess.run(input_learn)
    output_learn_run = sess.run(output_learn,
                              feed_dict={input_learn: input_learn_run})
    sess.close()
    
    return input_learn_run, output_learn_run

#Placeholder for the input and output are defined:
x = tf.placeholder(tf.float32, [None, nfeatures])
y = tf.placeholder(tf.float32, [None, 1])

#Here all parameters are defined:
tf.set_random_seed(1735)

#Weights between input and hidden layer nr. 1 with 10 nodes:
W1 = tf.Variable(tf.random_uniform([nfeatures, 10], minval=-0.6794, maxval=0.6794), name='W1')
#Bias of hidden-layer nr. 1:
b1 = tf.Variable(tf.random_normal([10], mean=0, stddev=0), name='b1')
#Weights between hidden layer nr. 1 and hidden layer nr. 2 with 10 nodes:
W2 = tf.Variable(tf.random_uniform([10, 10], minval=-0.5477, maxval=0.5477), name='W2')
#Bias of hidden-layer nr. 2:
b2 = tf.Variable(tf.random_normal([10], mean=0, stddev=0), name='b2')
#Weights between hidden layer nr. 2 and hidden layer nr. 3 with 10 nodes:
W3 = tf.Variable(tf.random_uniform([10, 10], minval=-0.5477, maxval=0.5477), name='W3')
#Bias of hidden-layer nr. 3:
b3 = tf.Variable(tf.random_normal([10], mean=0, stddev=0), name='b3')
#Weights between hidden layer nr. 3 and output:
W4 = tf.Variable(tf.random_uniform([10, 1], minval=-0.7385, maxval=0.7385), name='W4')
#Bias of the outputs:
b4 = tf.Variable(tf.random_normal([1], mean=0, stddev=0), name='b4')

#Here all the state variables are calculated:
#Calculating the values of hidden-layer nr.1:
hidden_1_out = tf.add(tf.matmul(x, W1), b1)
hidden_1_out = tf.nn.relu(hidden_1_out)
#Calculating the values of hidden-layer nr.2:
hidden_2_out = tf.add(tf.matmul(hidden_1_out, W2), b2)
hidden_2_out = tf.nn.relu(hidden_2_out)
#Calculating the values of hidden-layer nr.3:
hidden_3_out = tf.add(tf.matmul(hidden_2_out, W3), b3)
hidden_3_out = tf.nn.relu(hidden_3_out)
#Calculating the output: 
y_pred = tf.add(tf.matmul(hidden_3_out, W4), b4)

#The format of loss_minibatch is choosen to work with the function train_minibatch().
loss_minibatch = tf.reduce_mean(tf.squared_difference(y, y_pred))

#Here the empirical risk is determined:
x_test = tf.placeholder(tf.float32, [None, nfeatures])
y_test = tf.placeholder(tf.float32, [None, 1])
#Calculating the values of hidden-layer nr.1:
hidden_1_out_test = tf.add(tf.matmul(x_test, W1), b1)
hidden_1_out_test = tf.nn.relu(hidden_1_out_test)
#Calculating the values of hidden-layer nr.2:
hidden_2_out_test = tf.add(tf.matmul(hidden_1_out_test, W2), b2)
hidden_2_out_test = tf.nn.relu(hidden_2_out_test)
#Calculating the values of hidden-layer nr.3:
hidden_3_out_test = tf.add(tf.matmul(hidden_2_out_test, W3), b3)
hidden_3_out_test = tf.nn.relu(hidden_3_out_test)
#Calculating the output: 
y_pred_test = tf.add(tf.matmul(hidden_3_out_test, W4), b4)

#Calculating the empirical risk:
risk = tf.reduce_mean(tf.squared_difference(y_test, y_pred_test))

def estimate_risk(test_size, sess, j, stddev):
    #j is used as a seed, for creating different testing data
    #when estimate_risk is used multiple times.
    #testing data:
    #input:
    input_test = tf.random_uniform([test_size, nfeatures], minval=0, maxval=100, name="input_test", seed=j)

    #output:
    factors_test = tf.constant([[2], [-5], [4]], dtype="float32")
    norm_test = tf.random_normal([test_size, 1], mean=0, stddev=stddev, seed=2*j)
    output_test3 = tf.linalg.matmul(input_test, factors_test)
    output_test2 = tf.math.add(output_test3, tf.constant(30.0, shape=[test_size, 1]))
    output_test = tf.math.add(output_test2, norm_test, name="output_test")

    input_test_run = sess.run(input_test)
    output_test_run = sess.run(output_test,
                              feed_dict={input_test: input_test_run})

    risk_est = sess.run(risk,
                        feed_dict={x_test: input_test_run,
                               y_test: output_test_run})
    return risk_est

#The euclidean norm of the gradienten is calculated:
def compute_grad_norm(a):
    #The entries in a are inserted in l which is a vector whose norm is measureable:
    l = []
    for i in range(len(a)):
        shape = a[i].shape
        for k1 in range(shape[0]):
            if len(shape) == 2:                
                for k2 in range(shape[1]):
                    l.append(a[i][k1][k2])
            else:
                l.append(a[i][k1])
    
    #Now the norm of l is measured:
    sess = tf.InteractiveSession()
    grad_norm = sess.run(tf.norm(l, 'euclidean'))
    sess.close()
    
    return grad_norm, l



#The euclidean norm of the difference between two variables-vectors is calculated:
def compute_change(tvs1, tvs2):
    #The differences of the entries in tvs2 and tvs1 are inserted in l 
    #which is a vector whose norm is measureable:
    l = []
    for i in range(len(tvs1)):
        shape = tvs1[i].shape
        for k1 in range(shape[0]):
            if len(shape) == 2:                
                for k2 in range(shape[1]):
                    l.append(tvs2[i][k1][k2] - tvs1[i][k1][k2])
            else:
                l.append(tvs2[i][k1] - tvs1[i][k1])
    
    #Now the norm of l is measured:
    sess = tf.InteractiveSession()
    change_norm = sess.run(tf.norm(l, 'euclidean'))
    sess.close()
    
    return change_norm



#The angle between two consecutive gradients a1 and a is calculated:
def compute_angle(a1, a):
    len_a1, l_a1 = compute_grad_norm(a1)
    len_a, l_a = compute_grad_norm(a)
    if len_a1 == 0 or len_a == 0:
        return 0.0
    
    sess = tf.InteractiveSession()
    prod = sess.run(tf.tensordot(l_a1, l_a, 1))
    prod2 = prod / (len_a1 * len_a)
    angle = sess.run(tf.math.acos(prod2)*180/m.pi)
    sess.close()
    #Due to inaccuracy during the calculation process 
    #values can be out of the theoretical possible range:
    if prod2 > 1:
        return 0.0
    if prod2 < -1:
        return 180.0
    return angle

#The function train_minibatch is based on the function with the same name
#from: http://deeplearnphysics.org/Blog/minibatch.html

def train_minibatch(opt_type, learning_rate, batch_size, minibatch_size, test_size,
                    ntests, stddev, distnr):
    #The number of epochs is calculated:
    nsteps = batch_size // minibatch_size
    #The optimization method is choosen:
    opt = opt_type(learning_rate)

    # 0) The trainable variables are called up:
    tvs = tf.trainable_variables()
    # 1) accum_vars is a placeholder to accumulate the gradient:
    accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
    # 2) Operation to assign the values in accum_vars to 0:
    zero_ops  = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    # 3) Operation to calculate the gradient of the loss-function relating to a minibatch:
    gvs = opt.compute_gradients(loss=loss_minibatch, var_list=tvs)
    # 4) Operation to accumulate the gradients in accum_vars:
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    #Note: The different gv[0] contain the values of the gradient.
    # 5) Operation to apply the gradient with opt to the parameters:
    apply_ops = opt.apply_gradients([(accum_vars[i], tv) for i, tv 
                                     in enumerate(tf.trainable_variables())])
    
    # Array for saving the risk values:
    risk_log = np.zeros(ntests)
    # Array for saving the L2-norm values of the gradient:
    grad_norm_log = np.zeros(ntests)
    # Array for saving the values of the angles between two consecutive gradients:
    angle_log = np.zeros(ntests)
    # Array for saving the L2-norm values of the difference between the parameter vectors:
    change_log = np.zeros(ntests)
    # Array for saving the distances of the parameter vectors to the one at distnr:
    dist_log = np.zeros(ntests - distnr)
    
    # The session gets started:
    sess = tf.InteractiveSession()
    # Initial values for the variables are created:
    sess.run(tf.global_variables_initializer())
    #input and output are executed for being applyable in feed_dict:
    input_learn_run, output_learn_run = create_data(batch_size, stddev)    
    # For a1 an initial value is set because otherwise an error would occur in the first epoch.
    sess.run(zero_ops)
    a1 = sess.run(accum_vars)
    print("initial values of the parameters:\n", sess.run(tvs),"\n\n")
    
    # j runs through the values from 0 to nsteps - (nsteps // ntests)
    # because in the last (nsteps // ntests) - 1 epochs 
    # no measurements are made.
    # Training time:
    for j in range(nsteps - (nsteps // ntests) + 1):
        sess.run(zero_ops)
        sess.run(accum_ops,
                  feed_dict={x: input_learn_run[j*minibatch_size:(j+1)*minibatch_size],
                             y: output_learn_run[j*minibatch_size:(j+1)*minibatch_size]})
        
        #Because of the if loop only in certain epochs 
        #the risk and the gradients norm is estimated:
        if j % (nsteps // ntests) == 0:
            #The risk is estimated:
            risk_log[j // (nsteps // ntests)] = estimate_risk(test_size, sess, j, stddev)
            #The euclidean norm of the gradient is calculated:
            a = sess.run(accum_vars)
            grad_norm_log[j // (nsteps // ntests)] = compute_grad_norm(a)[0]
            tvs_before_update = sess.run(tvs)
            #The angle between the gradient and the previous gradient is calculated:
            angle_log[j // (nsteps // ntests)] = compute_angle(a1, a)
            #As soon as the test with nr. distnr is executed 
            #the current parameters are saved:
            if j // (nsteps // ntests) == (distnr - 1):
                tvs_fix = sess.run(tvs)
            #Since the test with nr. distnr the distance to the current parameters
            #to tvs_fix is calculated.
            if j // (nsteps // ntests) > (distnr - 1):
                dist_log[j // (nsteps // ntests) - distnr] = compute_change(tvs_fix, sess.run(tvs))            
            
            #The number of tests is printed to show the simulations progress:
            print(j // (nsteps // ntests) + 1)
        
        #Because of the if loop only in certain epochs the gradient is saved
        #for the calculation of the angle in the next epoch:
        if (j+1) % (nsteps // ntests) == 0:
            a1 = sess.run(accum_vars)
        
        sess.run(apply_ops)

        if j % (nsteps // ntests) == 0:
            tvs_after_update = sess.run(tvs)
            change_log[j // (nsteps // ntests)] = compute_change(tvs_before_update, tvs_after_update)

    print("gradient of the last epoch:\n", a, "\n\n")
    tvs_run = sess.run(tvs)    
    sess.close()
    return risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log

#The function train_minibatch_diminish is based on the function train_minibatch
#from: http://deeplearnphysics.org/Blog/minibatch.html

def train_minibatch_diminish(opt_type, learning_rate, batch_size, minibatch_size, test_size,
                             ntests, stddev, distnr):
    #The number of epochs is calculated:
    nsteps = batch_size // minibatch_size
    #The optimization method is choosen:
    lr = tf.placeholder(tf.float32)
    opt = opt_type(lr)

    # 0) The trainable variables are called up:
    tvs = tf.trainable_variables()
    # 1) accum_vars is a placeholder to accumulate the gradient:
    accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
    # 2) Operation to assign the values in accum_vars to 0:
    zero_ops  = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    # 3) Operation to calculate the gradient of the loss-function relating to a minibatch:
    gvs = opt.compute_gradients(loss=loss_minibatch, var_list=tvs)
    # 4) Operation to accumulate the gradients in accum_vars:
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
     #Note: The different gv[0] contain the values of the gradient.
    # 5) Operation to apply the gradient with opt to the parameters:
    apply_ops = opt.apply_gradients([(accum_vars[i], tv) for i, tv 
                                     in enumerate(tf.trainable_variables())])
    
    # Array for saving the risk values:
    risk_log = np.zeros(ntests)
    # Array for saving the L2-norm values of the gradient:
    grad_norm_log = np.zeros(ntests)
    # Array zum speichern der Werte der Winkel zwischen zwei aufeinanderfolgen Gradienten:
    angle_log = np.zeros(ntests)
    # Array for saving the L2-norm values of the difference between the parameter vectors:
    change_log = np.zeros(ntests)
    # Array for saving the distances of the parameter vectors to the one at distnr:
    dist_log = np.zeros(ntests - distnr)
    
    # The session gets started:
    sess = tf.InteractiveSession()
    # Initial values for the variables are created:
    sess.run(tf.global_variables_initializer())
    #input and output are executed for being applyable in feed_dict:
    input_learn_run, output_learn_run = create_data(batch_size, stddev)    
    # For a1 an initial value is set because otherwise an error would occur in the first epoch.
    sess.run(zero_ops)
    a1 = sess.run(accum_vars)
    print("initial values of the parameters:\n", sess.run(tvs),"\n\n")
    
    # j runs through the values from 0 to nsteps - (nsteps // ntests)
    # because in the last (nsteps // ntests) - 1 epochs 
    # no measurements are made.
    # Training time:
   for j in range(nsteps - (nsteps // ntests) + 1):
        sess.run(zero_ops)
        sess.run(accum_ops,
                  feed_dict={x: input_learn_run[j*minibatch_size:(j+1)*minibatch_size],
                             y: output_learn_run[j*minibatch_size:(j+1)*minibatch_size],
                             lr: learning_rate/(j+1)})
        
        #Because of the if loop only in certain epochs 
        #the risk and the gradients norm is estimated:
        if j % (nsteps // ntests) == 0:
            #The risk is estimated:
            risk_log[j // (nsteps // ntests)] = estimate_risk(test_size, sess, j, stddev)
            #The euclidean norm of the gradient is calculated:
            a = sess.run(accum_vars)
            grad_norm_log[j // (nsteps // ntests)] = compute_grad_norm(a)[0]
            tvs_before_update = sess.run(tvs)
            #The angle between the gradient and the previous gradient is calculated:
            angle_log[j // (nsteps // ntests)] = compute_angle(a1, a)
            #As soon as the test with nr. distnr is executed 
            #the current parameters are saved:
            if j // (nsteps // ntests) == (distnr - 1):
                tvs_fix = sess.run(tvs)
            #Since the test with nr. distnr the distance to the current parameters
            #to tvs_fix is calculated.
            if j // (nsteps // ntests) > (distnr - 1):
                dist_log[j // (nsteps // ntests) - distnr] = compute_change(tvs_fix, sess.run(tvs))

            #The number of tests is printed to show the simulations progress:
            print(j // (nsteps // ntests) + 1)

        #Because of the if loop only in certain epochs the gradient is saved
        #for the calculation of the angle in the next epoch:
        if (j+1) % (nsteps // ntests) == 0:
            a1 = sess.run(accum_vars)
        
        sess.run(apply_ops,
                feed_dict={lr: learning_rate/(j+1)})
        
        if j % (nsteps // ntests) == 0:
            tvs_after_update = sess.run(tvs)
            change_log[j // (nsteps // ntests)] = compute_change(tvs_before_update, tvs_after_update)

    print("gradient of the last epoch:\n", a, "\n\n")
    tvs_run = sess.run(tvs)    
    sess.close()
    return risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log

#The function Simulation() is based on the function "compare()"
#from: http://deeplearnphysics.org/Blog/minibatch.html

# The following learning options could be inserted for opt_type:
# "Stochastic Gradient Descent":  tf.train.GradientDescentOptimizer
# "Adagrad":                           tf.train.AdagradOptimizer
# "Adam":                              tf.train.AdamOptimizer
# "RMSProp"                            tf.train.RMSPropOptimizer

def Simulation(name, opt_type, learning_rate, batch_size, minibatch_size, test_size, ntests,
               stddev, diminish, distnr=1):
    # Minibatch-training is executed:
    if diminish == False:
        #In this case the training is executed with a constant learning rate:
        risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log = train_minibatch (opt_type=opt_type, 
                                             learning_rate=learning_rate,
                                             batch_size=batch_size,
                                             minibatch_size=minibatch_size,
                                             test_size=test_size,
                                             ntests=ntests,
                                             stddev=stddev,
                                             distnr=distnr)
    if diminish == True:
        #In this case the training is executed with a diminishing learning rate:
        risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log = train_minibatch_diminish (opt_type=opt_type, 
                                             learning_rate=learning_rate,
                                             batch_size=batch_size,
                                             minibatch_size=minibatch_size,
                                             test_size=test_size,
                                             ntests=ntests,
                                             stddev=stddev,
                                             distnr=distnr)
        
    print("final values of the parameters:\n", tvs_run, "\n\n")

    sess = tf.InteractiveSession()

    print("empirical risk:\n", risk_log)
    # In the graph the natural logarithm of the empirical risk is shown.
    plt.plot(range(1, ntests+1), sess.run(tf.math.log(risk_log)), linestyle="", marker="o", markersize=3)
    plt.ylabel("ln(Empirical Risk)")
    plt.xlabel("Measurement Nr.")
    plt.title(name)
    plt.grid(True)
    plt.show()        

    print("L2 norm of the gradients:\n", grad_norm_log)
    # In the graph the natural logarithm of the gradients norm is shown.
    plt.plot(range(1, ntests+1), sess.run(tf.math.log(grad_norm_log)), linestyle="", marker="o", markersize=3)
    plt.ylabel("ln(L2 Norm Of The Gradient)")
    plt.xlabel("Measurement Nr.")
    plt.title(name)
    plt.grid(True)
    plt.show()
    
    print("L2 norm of the difference between two consecutive parameter vectors:\n", change_log)
    # In the graph the natural logarithm of the differences norm is shown.
    plt.plot(range(1, ntests+1), sess.run(tf.math.log(change_log)), linestyle="", marker="o", markersize=3)
    plt.ylabel("ln(L2 Norm Of The Difference Between Two Consecutive Parameter Vectors)")
    plt.xlabel("Measurement Nr.")
    plt.title(name)
    plt.grid(True)
    plt.show()
    
    # The first value of angle_log is not printed
    # because an angle cannot be measured in the first epoch.
    print("angle between two consecutive parameter vectors:\n", angle_log[1:test_size])
    plt.plot(range(1, ntests), angle_log[1:test_size], linestyle="", marker="o", markersize=3)
    plt.ylabel("Angles Between Two Consecutive Parameter Vectors")
    plt.xlabel("Measurement Nr.")
    plt.title(name)
    plt.grid(True)
    plt.show()
 
    print("distance between the parameters and the parameters at epoch ", distnr, ":\n", dist_log)
    plt.plot(range(distnr+1, ntests+1), dist_log, linestyle="", marker="o", markersize=3)
    plt.ylabel("Distance Between The Parameters And \n The Parameters At Epoch " + str(distnr))
    plt.xlabel("Measurement Nr.")
    plt.title(name)
    plt.grid(True)
    plt.show()

    sess.close()