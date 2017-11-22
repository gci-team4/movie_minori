#coding: utf-8
import numpy as np
import pylab
import math
from sklearn.datasets import fetch_mldata #for loading dataset


num_of_output = 10
num_of_hidden_neuron = 500
num_of_hidden_layer = 1

#how much mahine learn(0<,<1)
theta = 0.03

def func(x):
    return 1 / (1+math.exp(-x))

def dif_func(x):
    return x * (1-x)

def mean_square(error):
    square = error*error
    return square.sum()/len(error)

def weigh(input,wih):
    hidden = np.ones(len(wih))
    for j in range(len(wih)):
        product = wih[j]*input
        #        print func(product.sum())
        hidden[j] = func(product.sum())
    return hidden

def calc_error_hidden(error_output,who,hidden):
    error_hidden = np.zeros(len(hidden))
    for j in range(len(hidden)):
        product = error_output*who.T[j]*dif_func(hidden)[j]
#        print "product = "
#        print product
        error_hidden[j] = product.sum()
    return error_hidden

def show(x):
    #show data for test
    for i in range(1,len(x)):
        if x[i]==0.0: print "□",
        else : print "■",
        if (i)%28==0: print"\n"
    
    return


def main():
    #download mnist
    mnist = fetch_mldata('MNIST original', data_home=".")

    #make traning dataset
    tr_data = mnist.data
    tr_label = mnist.target
    
    #set data to 0-1.0
    tr_data = tr_data.astype(np.float64)
    tr_data /=tr_data.max()
    
    #initialize weight
    #weight matrix has avirtual cell for bias in [0]
    #wih-> weight input to hidden layer, who-> weight hidden layer to output
        wih = np.random.random([num_of_hidden_neuron+1,tr_data.shape[1]+1])*0.01
    who = np.random.random([num_of_output,num_of_hidden_neuron+1])*0.01
    
    
    #neuron vector has a virtual cell(=1) for bias in [0]
    one_for_input = np.ones(tr_data.shape[0])
    tr_data = np.hstack((np.reshape(one_for_input,(-1,1)),tr_data))
    
#    #initialize hidden layer
#    hidden = np.ones((num_of_output,num_of_hidden_layer+1))

    """
    bias_hidden = np.ones(num_of_hidden_neuron)
    bias_output = np.ones(num_of_output)
    """
    """
    #initialize errors
    error_hidden = np.ones(num_of_hidden_neuron)
    error_output = np.ones(num_of_output)
    """
    
    #fix type of teacher to vector
    teacher = np.zeros([len(tr_label),10])
    for i in range(len(tr_label)):
        teacher[i][tr_label[i]]= 1

    random_int = np.arange(60000)
    np.random.shuffle(random_int)
    
    #calc for each training-data
    for data_num in range(60000):
        show(tr_data[random_int[data_num]])
#        print "wih="
#        print wih
#        print "who="
#        print who
#        print tr_label[random_int[data_num]]
        hidden = weigh(tr_data[random_int[data_num]],wih) #calc hidden layer
#        print "hidden is"
#        print hidden
        output = weigh(hidden,who) #calc output layer
#        print "output is"
#        print output

#        print "pre error_output = "
#        print teacher[random_int[data_num]] - output
        error_output = (teacher[random_int[data_num]] - output) * dif_func(output) #calc output error
#        print "error_output is"
#        print error_output
#        print mean_square(error_output)

        print mean_square(error_output)

        if mean_square(error_output) < 0.00001 : break
        
        #        print who.shape
#        print "who.T is"
#        print who.T.shape

        error_hidden = calc_error_hidden(error_output,who,hidden)
        
#        print "error_hidden = "
#        print error_hidden
#        
#        
#        print "who pre = "
#        print np.reshape(error_output,(-1,1)) * hidden

#        np.set_printoptions(threshold=np.inf)

#        print np.reshape(error_hidden,(-1,1))
#        print tr_data[random_int[data_num]]

#        print "wih pre = "
#        print np.reshape(error_hidden,(-1,1)) * tr_data[random_int[data_num]]

#        np.set_printoptions(threshold=7)

        #update weight
        who = who + theta * np.reshape(error_output,(-1,1)) * hidden
        wih = wih + theta * np.reshape(error_hidden,(-1,1)) * tr_data[random_int[data_num]]

    print data_num

    for i in range(60):
        input = tr_data[60000+i*100]
        hidden = weigh(input,wih) #calc hidden layer
        output = weigh(hidden,who) #calc output layer
        show(input)
        print output
        print np.argmax(output)
        print tr_label[60000+i*100]

    return

main()
#print type(traning)

#print tr_data.shape[1]





##draw 25 sample at random
#p = np.random.random_integers(0,len(mnist.data),25)
#for index,(data,label) in enumerate(np.array(zip(mnist.data,mnist.target))[p]):
#    pylab.subplot(5,5,index+1)
#    pylab.axis('off')
#    pylab.imshow(data.reshape(28,28),cmap=pylab.cm.gray_r,interpolation='nearest')
#    pylab.title('%i' % label)
#pylab.show()
#
