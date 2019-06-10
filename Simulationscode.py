import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import math as m
import matplotlib.pyplot as plt
%matplotlib inline

#nfeatures ist die Dimension des Inputs.
nfeatures = 3

#Durch den festgelegten seed werden immer die gleichen Daten erzeugt.
#Wählt man als batch_size einmal die Zahl n und ein anderes mal die Zahl n+1,
#So gleichen sich jeweils die ersten n Daten.
def create_data(batch_size, stddev):
    #Parameter für die Dimension des Inputs:
    
    #Hier werden die Daten generiert:
    #Input:
    input_learn = tf.random_uniform([batch_size, nfeatures], minval=0, maxval=100, seed=3140, name="input")

    #Output:
    factors = tf.constant([[2], [-5], [4]], dtype="float32")
    norm = tf.random_normal([batch_size, 1], mean=0, stddev=stddev, seed=8235)
    output_learn3 = tf.linalg.matmul(input_learn, factors)
    output_learn2 = tf.math.add(output_learn3, tf.constant(30.0, shape=[batch_size, 1]))
    output_learn = tf.math.add(output_learn2, norm, name="output")

    
    #Die Daten werden ausgeführt:
    sess = tf.InteractiveSession()
    input_learn_run = sess.run(input_learn)
    output_learn_run = sess.run(output_learn,
                              feed_dict={input_learn: input_learn_run})
    sess.close()
    
    return input_learn_run, output_learn_run

#Placeholder für Input und Output werden definiert:
x = tf.placeholder(tf.float32, [None, nfeatures])
y = tf.placeholder(tf.float32, [None, 1])

#Hier werden alle Parameter definiert:
tf.set_random_seed(1735)

# Gewichte zwischen dem Input und hidden-layer nr. 1 mit 10 Knoten:
W1 = tf.Variable(tf.random_uniform([nfeatures, 10], minval=-0.6794, maxval=0.6794), name='W1')
#Bias von hidden-layer nr. 1:
b1 = tf.Variable(tf.random_normal([10], mean=0, stddev=0), name='b1')
# Gewichte zwischen hidden-layer nr.1 und hidden-layer nr. 2 mit 50 Knoten:
W2 = tf.Variable(tf.random_uniform([10, 10], minval=-0.5477, maxval=0.5477), name='W2')
#Bias von hidden-layer nr. 2:
b2 = tf.Variable(tf.random_normal([10], mean=0, stddev=0), name='b2')
# Gewichte zwischen hidden-layer nr.2 und hidden-layer nr. 3 mit 50 Knoten:
W3 = tf.Variable(tf.random_uniform([10, 10], minval=-0.5477, maxval=0.5477), name='W3')
#Bias von hidden-layer nr. 3:
b3 = tf.Variable(tf.random_normal([10], mean=0, stddev=0), name='b3')
# Gewichte zwischen hidden-layer nr.3 und dem Output:
W4 = tf.Variable(tf.random_uniform([10, 1], minval=-0.7385, maxval=0.7385), name='W4')
#Bias des Outputs:
b4 = tf.Variable(tf.random_normal([1], mean=0, stddev=0), name='b4')

# Hier werden alle Zustandsvariablen berechnet:
# Berechne die Werte von hidden-layer nr.1:
hidden_1_out = tf.add(tf.matmul(x, W1), b1)
hidden_1_out = tf.nn.relu(hidden_1_out)
# Berechne die Werte von hidden-layer nr.2:
hidden_2_out = tf.add(tf.matmul(hidden_1_out, W2), b2)
hidden_2_out = tf.nn.relu(hidden_2_out)
# Berechne die Werte von hidden-layer nr.3:
hidden_3_out = tf.add(tf.matmul(hidden_2_out, W3), b3)
hidden_3_out = tf.nn.relu(hidden_3_out)
# Berechne den Output: 
y_pred = tf.add(tf.matmul(hidden_3_out, W4), b4)

#Das Format von loss_minibatch ist für die Anwandung in der Funktion train_minibatch() zugeschnitten.
loss_minibatch = tf.reduce_mean(tf.squared_difference(y, y_pred))

#Hier wird das Risiko empirisch bestimmt:
x_test = tf.placeholder(tf.float32, [None, nfeatures])
y_test = tf.placeholder(tf.float32, [None, 1])
# Berechne die Werte von hidden-layer nr.1:
hidden_1_out_test = tf.add(tf.matmul(x_test, W1), b1)
hidden_1_out_test = tf.nn.relu(hidden_1_out_test)
# Berechne die Werte von hidden-layer nr.2:
hidden_2_out_test = tf.add(tf.matmul(hidden_1_out_test, W2), b2)
hidden_2_out_test = tf.nn.relu(hidden_2_out_test)
# Berechne die Werte von hidden-layer nr.3:
hidden_3_out_test = tf.add(tf.matmul(hidden_2_out_test, W3), b3)
hidden_3_out_test = tf.nn.relu(hidden_3_out_test)
# Berechne den Output:
y_pred_test = tf.add(tf.matmul(hidden_3_out_test, W4), b4)

#Berechne das empirische Risiko:
risk = tf.reduce_mean(tf.squared_difference(y_test, y_pred_test))

def estimate_risk(test_size, sess, j, stddev):
    #j wird als seed verwendet, damit bei verschiedenen estimate_risk
    #verschiedene Testdaten erzeugt werden
    #Testdaten:
    #Input:
    input_test = tf.random_uniform([test_size, nfeatures], minval=0, maxval=100, name="input_test", seed=j)

    #Output:
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

#Die euklidische Norm des Gradienten wird berechnet:
def compute_grad_norm(a):
    #Die Eiträge von a werden in l eingefügt,
    #wobei l ein Vektor ist, dessen Norm sich messen läst:
    l = []
    for i in range(len(a)):
        shape = a[i].shape
        for k1 in range(shape[0]):
            if len(shape) == 2:                
                for k2 in range(shape[1]):
                    l.append(a[i][k1][k2])
            else:
                l.append(a[i][k1])
    
    #Nun wird die Norm von l gemessen:
    sess = tf.InteractiveSession()
    grad_norm = sess.run(tf.norm(l, 'euclidean'))
    sess.close()
    
    return grad_norm, l



#Die euklidische Norm der Differenz zweier Variablen-Vektoren wird berechnet:
def compute_change(tvs1, tvs2):
    #In l werden die Differenzen der Einträge von tvs2 und tvs1 eigefügt,
    #wobei l ein Vektor ist, dessen Norm sich messen läst:
    l = []
    for i in range(len(tvs1)):
        shape = tvs1[i].shape
        for k1 in range(shape[0]):
            if len(shape) == 2:                
                for k2 in range(shape[1]):
                    l.append(tvs2[i][k1][k2] - tvs1[i][k1][k2])
            else:
                l.append(tvs2[i][k1] - tvs1[i][k1])
    
    #Nun wird die Norm von l gemessen:
    sess = tf.InteractiveSession()
    change_norm = sess.run(tf.norm(l, 'euclidean'))
    sess.close()
    
    return change_norm



#Der Winkel zwischen zwei aufeinanderfolgen Gradienten a1 und a wird berechnet:
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
    #Wegen Rechenungenauigkeiten kann prod2 außerhalb der
    #theoretisch möglichen Range liegen.
    if prod2 > 1:
        return 0.0
    if prod2 < -1:
        return 180.0
    return angle

#Die Funktion train_minibatch orientiert sich an der gleichnamigen Funktion 
#von: http://deeplearnphysics.org/Blog/minibatch.html

def train_minibatch(opt_type, learning_rate, batch_size, minibatch_size, test_size,
                    ntests, stddev, distnr):
    #Die Anzahl der Durchgänge wird berechnet:
    nsteps = batch_size // minibatch_size
    # Das Optimierungsverfahren wird festgelegt:
    opt = opt_type(learning_rate)

    # 0) Die trainierbaren Variablen werden abgerufen:
    tvs = tf.trainable_variables()
    # 1) accum_vars ist ein placeholder für die Akkumulation des Gradienten:
    accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
    # 2) Operation um die Werte von accum_vars auf 0 zu setzen:
    zero_ops  = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    # 3) Operation um den Gradienten der loss-function bezüglich eines minibatch zu berechnen:
    gvs = opt.compute_gradients(loss=loss_minibatch, var_list=tvs)
    # 4) Operation um die Gradienten in accum_vars zu akkumulieren:
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    #Anmerkung: Die verschiedenen gv[0] enthalten die Zahlenwerte des Gradienten.
    # 5) Operation um den Gradienten gemäß opt auf die Parameter anzuwenden:
    apply_ops = opt.apply_gradients([(accum_vars[i], tv) for i, tv 
                                     in enumerate(tf.trainable_variables())])
    
    # Array zum speichern der Risiko-Werte:
    risk_log = np.zeros(ntests)
    # Array zum speichern der L2-Norm Werte der Gradienten:
    grad_norm_log = np.zeros(ntests)
    # Array zum speichern der Werte der Winkel zwischen zwei aufeinanderfolgen Gradienten:
    angle_log = np.zeros(ntests)
    # Array zum speichern der L2-Norm der differenz der Parametervektoren:
    change_log = np.zeros(ntests)
    # Array zum speichern der Abstände von:
    dist_log = np.zeros(ntests - distnr)
    
    # Die session wird gestartet:
    sess = tf.InteractiveSession()
    # Startwerte für die Variblen werden erzeugt:
    sess.run(tf.global_variables_initializer())
    #input und output werden ausgeführt, um in feed_dict verwendbar zu sein:
    input_learn_run, output_learn_run = create_data(batch_size, stddev)    
    # a1 erhält einen Startwert, damit im ersten Durchgang kein Error auftritt.
    sess.run(zero_ops)
    a1 = sess.run(accum_vars)
    print("Startwerte der Parameter:\n", sess.run(tvs),"\n\n")
    
    # j nimmt die Werte von 0 bis nsteps - (nsteps // ntests) an,
    # da in den letzten (nsteps // ntests) - 1 Durchgängen 
    # keine Messungen mehr durchgeführt werden.
    # Trainings Phase:
    for j in range(nsteps - (nsteps // ntests) + 1):
        sess.run(zero_ops)
        sess.run(accum_ops,
                  feed_dict={x: input_learn_run[j*minibatch_size:(j+1)*minibatch_size],
                             y: output_learn_run[j*minibatch_size:(j+1)*minibatch_size]})
        
        #Die if Abfrage sorgt dafür, dass nur in bestimmten Schritten das 
        #Risiko und die Gradientennorm geschätzt werden:
        if j % (nsteps // ntests) == 0:
            #Das Risiko wird geschätzt:
            risk_log[j // (nsteps // ntests)] = estimate_risk(test_size, sess, j, stddev)
            #Die euklidische Norm des Gradienten wird berechnet:
            a = sess.run(accum_vars)
            grad_norm_log[j // (nsteps // ntests)] = compute_grad_norm(a)[0]
            tvs_before_update = sess.run(tvs)
            #Der Winkel zwischen dem Gradienten und dem vorherigen Gradienten wird berechnet:
            angle_log[j // (nsteps // ntests)] = compute_angle(a1, a)
            #Sobald der Test der Nummer distnr durchgeführt wird, werden die 
            #aktuellen Parameter gespeichert:
            if j // (nsteps // ntests) == (distnr - 1):
                tvs_fix = sess.run(tvs)
            #Ab dem Test der Nummer distnr wird der Abstand der aktuellen Parameter
            #zu tvs_fix berechnet.
            if j // (nsteps // ntests) > (distnr - 1):
                dist_log[j // (nsteps // ntests) - distnr] = compute_change(tvs_fix, sess.run(tvs))            
            
            #Die Nummer des Tests wird ausgegeben, um den Simulationsfortschritt anzuzeigen:
            print(j // (nsteps // ntests) + 1)
        
        #Die if Abfrage sorgt dafür, dass nur in bestimmten Schritten der Gradient 
        #für die Winkelberechnung im nächsten Durchgang gespeichert wird:
        if (j+1) % (nsteps // ntests) == 0:
            a1 = sess.run(accum_vars)
        
        sess.run(apply_ops)

        if j % (nsteps // ntests) == 0:
            tvs_after_update = sess.run(tvs)
            change_log[j // (nsteps // ntests)] = compute_change(tvs_before_update, tvs_after_update)

    print("Gradient des letzten Durchgangs:\n", a, "\n\n")
    tvs_run = sess.run(tvs)    
    sess.close()
    return risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log

#Die Funktion train_minibatch_diminish orientiert sich an der gleichnamigen Funktion 
#von: http://deeplearnphysics.org/Blog/minibatch.html

def train_minibatch_diminish(opt_type, learning_rate, batch_size, minibatch_size, test_size,
                             ntests, stddev, distnr):
    #Die Anzahl der Durchgänge wird berechnet:
    nsteps = batch_size // minibatch_size
    # Das Optimierungsverfahren wird festgelegt:
    lr = tf.placeholder(tf.float32)
    opt = opt_type(lr)

    # 0) Die trainierbaren Variablen werden abgerufen:
    tvs = tf.trainable_variables()
    # 1) accum_vars ist ein placeholder für die Akkumulation des Gradienten:
    accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
    # 2) Operation um die Werte von accum_vars auf 0 zu setzen:
    zero_ops  = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    # 3) Operation um den Gradienten der loss-function bezüglich eines minibatch zu berechnen:
    gvs = opt.compute_gradients(loss=loss_minibatch, var_list=tvs)
    # 4) Operation um die Gradienten in accum_vars zu akkumulieren:
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    #Anmerkung: Die verschiedenen gv[0] enthalten die Zahlenwerte des Gradienten.
    # 5) Operation um den Gradienten gemäß opt auf die Parameter anzuwenden:
    apply_ops = opt.apply_gradients([(accum_vars[i], tv) for i, tv 
                                     in enumerate(tf.trainable_variables())])
    
    # Array zum speichern der Risiko-Werte:
    risk_log = np.zeros(ntests)
    # Array zum speichern der L2-Norm Werte der Gradienten:
    grad_norm_log = np.zeros(ntests)
    # Array zum speichern der Werte der Winkel zwischen zwei aufeinanderfolgen Gradienten:
    angle_log = np.zeros(ntests)
    # Array zum speichern der L2-Norm der differenz der Parametervektoren:
    change_log = np.zeros(ntests)
    # Array zum speichern der Abstände von:
    dist_log = np.zeros(ntests - distnr)
    
    # Die session wird gestartet:
    sess = tf.InteractiveSession()
    # Startwerte für die Variblen werden erzeugt:
    sess.run(tf.global_variables_initializer())
    #input und output werden ausgeführt, um in feed_dict verwendbar zu sein:
    input_learn_run, output_learn_run = create_data(batch_size, stddev)    
    # a1 erhält einen Startwert, damit im ersten Durchgang kein Error auftritt.
    sess.run(zero_ops)
    a1 = sess.run(accum_vars)
    print("Startwerte der Parameter:\n", sess.run(tvs),"\n\n")
    
    # j nimmt die Werte von 0 bis nsteps - (nsteps // ntests) an,
    # da in den letzten (nsteps // ntests) - 1 Durchgängen 
    # keine Messungen mehr durchgeführt werden.
    # Trainings Phase:
    for j in range(nsteps - (nsteps // ntests) + 1):
        sess.run(zero_ops)
        sess.run(accum_ops,
                  feed_dict={x: input_learn_run[j*minibatch_size:(j+1)*minibatch_size],
                             y: output_learn_run[j*minibatch_size:(j+1)*minibatch_size],
                             lr: learning_rate/(j+1)})
        
        #Die if Abfrage sorgt dafür, dass nur in bestimmten Schritten das 
        #Risiko und die Gradientennorm geschätzt werden:
        if j % (nsteps // ntests) == 0:
            #Das Risiko wird geschätzt:
            risk_log[j // (nsteps // ntests)] = estimate_risk(test_size, sess, j, stddev)
            #Die euklidische Norm des Gradienten wird berechnet:
            a = sess.run(accum_vars)
            grad_norm_log[j // (nsteps // ntests)] = compute_grad_norm(a)[0]
            tvs_before_update = sess.run(tvs)
            #Der Winkel zwischen dem Gradienten und dem vorherigen Gradienten wird berechnet:
            angle_log[j // (nsteps // ntests)] = compute_angle(a1, a)
            #Sobald der Test der Nummer distnr durchgeführt wird, werden die 
            #aktuellen Parameter gespeichert:
            if j // (nsteps // ntests) == (distnr - 1):
                tvs_fix = sess.run(tvs)
            #Ab dem Test der Nummer distnr wird der Abstand der aktuellen Parameter
            #zu tvs_fix berechnet.
            if j // (nsteps // ntests) > (distnr - 1):
                dist_log[j // (nsteps // ntests) - distnr] = compute_change(tvs_fix, sess.run(tvs))

                #Die Nummer des Tests wird ausgegeben, um den Simulationsfortschritt anzuzeigen:
            print(j // (nsteps // ntests) + 1)

        #Die if Abfrage sorgt dafür, dass nur in bestimmten Schritten der Gradient 
        #für die Winkelberechnung im nächsten Durchgang gespeichert wird:
        if (j+1) % (nsteps // ntests) == 0:
            a1 = sess.run(accum_vars)
        
        sess.run(apply_ops,
                feed_dict={lr: learning_rate/(j+1)})
        
        if j % (nsteps // ntests) == 0:
            tvs_after_update = sess.run(tvs)
            change_log[j // (nsteps // ntests)] = compute_change(tvs_before_update, tvs_after_update)

    print("Gradient des letzten Durchgangs:\n", a, "\n\n")
    tvs_run = sess.run(tvs)    
    sess.close()
    return risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log

#Die Funktion Simulation() orientiert sich an der Funktion "compare()"
#von: http://deeplearnphysics.org/Blog/minibatch.html

# Die folgenden Lernverfahren lassen sich beispielsweise für opt_type einsetzen:
# "Stochastischer Gradientenabstieg":  tf.train.GradientDescentOptimizer
# "Adagrad":                           tf.train.AdagradOptimizer
# "Adam":                              tf.train.AdamOptimizer
# "RMSProp"                            tf.train.RMSPropOptimizer

def Simulation(name, opt_type, learning_rate, batch_size, minibatch_size, test_size, ntests,
               stddev, diminish, distnr=1):
    # Minibatch-Training wird durchgeführt:
    if diminish == False:
        #In diesem Fall wird das Training mit konstanter Lernrate durchgeführt
        risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log = train_minibatch (opt_type=opt_type, 
                                             learning_rate=learning_rate,
                                             batch_size=batch_size,
                                             minibatch_size=minibatch_size,
                                             test_size=test_size,
                                             ntests=ntests,
                                             stddev=stddev,
                                             distnr=distnr)
    if diminish == True:
        #In diesem Fall wird das Training mit abnehmender Lernrate durchgeführt
        risk_log, grad_norm_log, change_log, tvs_run, angle_log, dist_log = train_minibatch_diminish (opt_type=opt_type, 
                                             learning_rate=learning_rate,
                                             batch_size=batch_size,
                                             minibatch_size=minibatch_size,
                                             test_size=test_size,
                                             ntests=ntests,
                                             stddev=stddev,
                                             distnr=distnr)
        
    print("Endwerte der Parameter:\n", tvs_run, "\n\n")

    sess = tf.InteractiveSession()

    print("Empirisches Risiko:\n", risk_log)
    # Im Graph wird der natürliche Logarithmus des empirische Risikos angezeigt.
    plt.plot(range(1, ntests+1), sess.run(tf.math.log(risk_log)), linestyle="", marker="o", markersize=3)
    plt.ylabel("ln(Empirisches Risiko)")
    plt.xlabel("Messnummer")
    plt.title(name)
    plt.grid(True)
    plt.show()        

    print("L2 Norm des Gradienten:\n", grad_norm_log)
    # Im Graph wird der natürliche Logarithmus der Norm des Gradienten angezeigt.
    plt.plot(range(1, ntests+1), sess.run(tf.math.log(grad_norm_log)), linestyle="", marker="o", markersize=3)
    plt.ylabel("ln(L2 Norm des Gradienten)")
    plt.xlabel("Messnummer")
    plt.title(name)
    plt.grid(True)
    plt.show()
    
    print("L2 Norm der Differenz zweier aufeinanderfolgender Parametervektoren:\n", change_log)
    # Im Graph wird der natürliche Logarithmus der Norm der Differenz angezeigt.
    plt.plot(range(1, ntests+1), sess.run(tf.math.log(change_log)), linestyle="", marker="o", markersize=3)
    plt.ylabel("ln(L2 Norm der Differenz zweier aufeinanderfolgender Parametervektoren)")
    plt.xlabel("Messnummer")
    plt.title(name)
    plt.grid(True)
    plt.show()
    
    # Der erste Wert von angle_log wird nicht ausgegeben,
    # da im ersten Durchgang noch kein Winkel gemessen werden konnte.
    print("Winkel zwischen zwei aufeinanderfolgenden Gradienten:\n", angle_log[1:test_size])
    plt.plot(range(1, ntests), angle_log[1:test_size], linestyle="", marker="o", markersize=3)
    plt.ylabel("Winkel zwischen zwei aufeinanderfolgenden Gradienten")
    plt.xlabel("Messnummer")
    plt.title(name)
    plt.grid(True)
    plt.show()
 
    print("Distanz zwischen Parametern und Parametern bei Messnumer ", distnr, ":\n", dist_log)
    plt.plot(range(distnr+1, ntests+1), dist_log, linestyle="", marker="o", markersize=3)
    plt.ylabel("Distanz zwischen Parametern und \n Parametern bei Messnumer " + str(distnr))
    plt.xlabel("Messnummer")
    plt.title(name)
    plt.grid(True)
    plt.show()

    sess.close()