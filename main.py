__author__ = 'shawn'

xrange = range
try:
    import PIL.Image as Image
except ImportError:
    import Image

import os
from RBM import *

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data




def get_data(file_name):
    f = open(file_name + '.save', 'rb')
    n_visable, n_hidden,  W, hbias, vbias = cPickle.load(f)
    f.close()
    return n_visable, n_hidden, to_shared(W), to_shared(hbias), to_shared(vbias)

from pylab import imshow, show, cm
import numpy as np
import scipy

def minipulate_and_save_image(dataset, index, size, precentage, n_repeat):
    """View a single image."""
    # dataset = dataset.get_value()
    data = []
    for current_size in range(0, size):
        image = np.array(dataset[index + current_size]).reshape(28,28)
        # imshow(image, cmap=cm.gray)
        # show()
        scipy.misc.imsave('origional_image' + str(current_size) + '.jpg', image)
        flip_counter = 0
        for (x,y), value in numpy.ndenumerate(image):
            if flip_value(precentage):
                flip_counter += 1
                image[x][y] = 1 - value
        scipy.misc.imsave('noise_image' + str(current_size) + '.jpg', image)
        dataset[index + current_size] = image.reshape(784)
        for i in range(0, n_repeat):
            data += [image.reshape(784)]
    print('flipped: ' + str(round((flip_counter / 784 ) * 100, 2)) + '%')
    return data


def flip_value(precentage):
    sample = np.random.random_integers(0,99)
    if sample < precentage:
        return True
    return False

def to_shared(v, borrow = True):
    return theano.shared(numpy.asarray(v, dtype=theano.config.floatX), borrow=borrow)


def train_rbm(rbm, dataset, learning_rate=0.1, training_epochs=10,
              batch_size=20, output_folder='rbm_plots', n_hidden=500, CD_steps=3):

    train_set_x, train_set_y = dataset

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # get the cost and the gradient corresponding to one step of CD-1
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=CD_steps)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()
    logFile = open('rbm_logs.txt', 'w')
    logFile.write('LR: ' + str(learning_rate) + ', Epoch: ' + str(training_epochs) + ', PCD-' + str(CD_steps) + '\n\n')
    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        logFile.write('Training epoch %d, cost is ' + str(numpy.mean(mean_cost)) + '\n' % epoch)

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    logFile.write('Training took %f minutes\n' % (pretraining_time / 60.))
    logFile.close()
    rbm.save_data('rbm_data')

def sample_rbm(rbm, test_set_x, n_chains=1, n_samples=100, n_step=1000, percentage_noise = 5,n_repeat = 1):
    rng = numpy.random.RandomState()
    if os.path.isdir('rbm_plots'):
        os.chdir('rbm_plots')
    if rbm == None:
        x = T.matrix('x')
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        n_visable, n_hidden, W, hbias, vbias = get_data('tmp')
        rbm = RBM(input=x, n_visible=n_visable, n_hidden=n_hidden, W=W,hbias=hbias, vbias=vbias ,numpy_rng=rng, theano_rng=theano_rng)

    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            minipulate_and_save_image(test_set_x.get_value(borrow=True),test_idx, n_chains, percentage_noise, n_repeat),
            dtype=theano.config.floatX
        )
    )


    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=n_step
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains * n_repeat - 1),
        dtype='uint8'
    )


    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        if idx * n_step < 25:
            print("normalizing")
            data = persistent_vis_chain.get_value()
            normal = test_normalize(data)
            data[len(data) -1] = test_set_x.get_value(borrow=True)[rng.randint(number_of_test_samples - n_chains)]
            persistent_vis_chain = theano.shared(
                numpy.asarray(
                    data,
                    dtype=theano.config.floatX
                )
            )

        print (' ... plotting sample ', idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains*n_repeat),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    # os.chdir('../')

def test_normalize(data):
    new_data = [0] * len(data[0])
    for x in range(0, len(data)):
        for y in range(0,len(data[x])):
            new_data[y] += data[x][y]
    new_data = [x / len(data[0]) for x in new_data]
    return new_data



n_hidden = 500
n_visable = 28 * 28
x = T.matrix('x')
datasets = load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[2]
rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
# rbm = RBM(input=x, n_visible=28 * 28,
#               n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)
# train_rbm(rbm, datasets[0], learning_rate=0.1, training_epochs=15,
#               batch_size=20, output_folder='rbm_plots', n_hidden=500, CD_steps=5)
sample_rbm(rbm=None, test_set_x=test_set_x, n_chains=1, n_samples=500, n_step=1,percentage_noise=5, n_repeat=10)