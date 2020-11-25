# EE40098 Computational Intelligence
# Ben Warwick-Champion

# Import scipy.special for the sigmoid function expit()
import scipy.special
import scipy.signal
import scipy as sp
import numpy
import matplotlib.pyplot as plt
import scipy.io as spio
#from statsmodels.nonparametric.smoothers_lowess import lowess

# Neural network class definition
class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)


    # Train the network using back-propagation
    def train(self, inputs_list, targets_list):
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T
        #Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets_array - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs_array))

    def query(self, inputs_list):
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # Save the neural network to a file
    def save(self, file_name, iterations=0):
        # Create an empty dictionary to store current state of neural network
        saved_network = {}
        # Store current state of neural network in dictionary
        saved_network['i_nodes'] = self.i_nodes
        saved_network['h_nodes'] = self.h_nodes
        saved_network['o_nodes'] = self.o_nodes
        saved_network['lr'] = self.lr
        saved_network['wih'] = self.wih
        saved_network['who'] = self.who
        saved_network['iterations'] = iterations   # Number of iterations at time save was done (defaults to 0)
        # Save dictionary to file
        spio.savemat(file_name, saved_network)

    # Load previously saved state from file
    def load(self, file_name):
        # Open saved file
        try:
            loaded_network = spio.loadmat(file_name)
            # Load data to correct arrays and variables
            self.i_nodes = loaded_network['i_nodes']
            self.h_nodes = loaded_network['h_nodes']
            self.o_nodes = loaded_network['o_nodes']
            self.lr = loaded_network['lr']
            self.wih = loaded_network['wih']
            self.who = loaded_network['who']
            iterations = loaded_network['iterations']
            return iterations

        except:
            # If file does not exist, return false
            return False
        
def peakDetection(input_samples, peak_height, peak_distance):
    # Uses the find_peaks function for the scipy.signal library to output an array of peaks.
    peaks_peak, _ = scipy.signal.find_peaks(input_samples, height=peak_height, \
    threshold=None, distance=peak_distance, prominence=None, width=None, \
    wlen=None, rel_height=0.5)

    # Initialise count and output array for the start point of each peak
    count = 0
    peaks_start = numpy.zeros(len(peaks_peak), dtype=int)
    # Iterate through each peak to find the start point
    for peak in peaks_peak:
        # Find the gradients of each point leading up to the peak
        gradient = numpy.gradient(input_samples[peak-50:peak])
        # Determine where the gradient changes sign
        asign = numpy.sign(gradient)
        signchange = ((numpy.roll(asign, 1) - asign) != 0).astype(int)
        # Start of the peak is the peak minus the highest non-zero indices
        peaks_start[count] = peak - numpy.argmax(numpy.nonzero(signchange))
        count += 1

    return peaks_start

def load_data(file_name):
    # Load the coursework training data
    mat = spio.loadmat(file_name, squeeze_me=True)
    dNoisy = mat['d']
    Index = mat['Index']
    Class = mat['Class']

    return dNoisy, Index, Class

def filter_data(noisy_input):
    #d = sp.signal.medfilt(dNoisy,11)
    #d = lowess(dNoisy, range(1440000), frac=0.2)
    # 3rd order butterworth filter for smoothing input data
    b, a = scipy.signal.butter(3, 0.2, btype='lowpass', analog=False)
    filtered_signal = scipy.signal.lfilter(b, a, noisy_input)

    return filtered_signal

def sort_data(Index, Class):
    # Sort 2 input lists maintaining their reference to each other
    tempIndex, tempClass = zip(*sorted(zip(Index, Class)))
    sortedIndex = numpy.array(tempIndex)
    sortedClass = numpy.array(tempClass)

    return sortedIndex, sortedClass

def scale_data(peaks_list):
    # Find the minimum value in the array of peaks
    min_val = min(peaks_list)
    # Shift the array up by the magnitude of this value to avoid negative values
    peaks_list = peaks_list + abs(min_val)
    # Get the maximum value of the array
    max_val = max(peaks_list)
    # Scale the array to be between 0.01 and 1
    peaks_list = (peaks_list / max_val) * 0.99 + 0.01

    return peaks_list

def plot_peaks(data, peaks_list):
    # Plots a graph of all the data indicating the start locations of a peak with an x.
    plt.plot(data)
    plt.plot(peaks_list, data[peaks_list], "x")
    plt.plot(numpy.zeros_like(data), "--", color="gray")
    plt.show()

def normalise_data(data_array):
    # 3rd order highpass butterworth filter to reduce the low frequency noise in data.
    b, a = scipy.signal.butter(3, 0.001, btype='highpass', analog=False)
    filtered_signal = scipy.signal.lfilter(b, a, data_array)

    return filtered_signal

def save_submission(save_file, d, Index, Class):
    submission_data = {}
    submission_data['d'] = d
    submission_data['Index'] = Index
    submission_data['Class'] = Class

    spio.savemat(save_file, submission_data)


def training_program(file_name, save_network=0, load_network=0, plot_output=0):
    # Import training data from file
    dNoisy, Index, Class = load_data(file_name)
    # Prepare data for analysis
    d = filter_data(dNoisy)
    sortedIndex, sortedClass = sort_data(Index, Class)

    # Find the peaks
    peaks = peakDetection(d, 2, 25)
    peaksDetected = len(peaks)
    peaksActual = len(Index)

    # Define the number of training loops
    train_loops = peaksActual

    # Scale data to prepare for input to NN
    d = scale_data(d)

    # Define neural network parameters
    input_nodes = 100
    hidden_nodes = 2*input_nodes
    output_nodes = 4
    learning_rate = 0.3
    # Define number of training iterations
    iterations = 10

    # Create a NeuralNetwork instance (nn) with the desired parameters
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Check whether a saved network needs to be loaded or to continue with a fresh one
    if load_network != 0:
        iterations = nn.load(load_network)
    else:
        for iteration in range(iterations):
            # Train the neural network on each training sample
            halfSampleSize = int(input_nodes/2)
            i = 0
            for sample in sortedIndex:
                print("\r Total Progress: " + str(round(100 * iteration / iterations)) + "%  ---  Current Iteration: " + str(
                        round((100 * i / train_loops), 1)) + "%", end='')
                inputs = numpy.asfarray(d[(sample-halfSampleSize):(sample+halfSampleSize)])
                targets = numpy.zeros(output_nodes) + 0.01
                # Incentivising the network to choose the right value
                targets[sortedClass[i]-1] = 0.99
                i += 1
                # Train the network
                nn.train(inputs, targets)

    # Scorecard list for how well the network performs, initially empty
    scorecard = []
    chosen_class = numpy.zeros(4)

    j = 0
    # Loop through all of the records in the test data set
    for sample in sortedIndex:
        # The correct label is the first value
        correct_label = int(sortedClass[j])
        j += 1

        print("Correct label: ", correct_label)
        # Scale and shift the inputs
        inputs = numpy.asfarray(d[sample-halfSampleSize:sample+halfSampleSize])
        # Query the network
        outputs = nn.query(inputs)
        # The index of the highest value output corresponds to the label
        label = numpy.argmax(outputs) + 1
        print("Network label: ", label)
        # Log which neuron was chosen
        chosen_class[label-1] += 1

        # Append either a 1 or 0 to the scorecard list
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    # Calculate the performance score, the fraction of the correct answers
    scorecard_array = numpy.asarray(scorecard)
    print("Performance = ", (scorecard_array.sum() / scorecard_array.size)*100, '%')

    if (scorecard_array.sum() / scorecard_array.size) > 0.982:
        nn.save('best.mat')
    if save_network != 0:
        nn.save(save_network)

    print("Number of peaks detected: ", peaksDetected)
    print("Actual number of peaks: ", peaksActual)
    for neuron_type in range(4):
        print("Type ", neuron_type+1, ": ", int(chosen_class[neuron_type]))

    # Plot data and detected peaks if requested
    if plot_output == 1:
        plot_peaks(d, peaks)

def submission_program(file_name, load_network=0, plot_output=0):
    # Import training data from file
    #dNoisy = load_data(file_name)
    mat = spio.loadmat(file_name, squeeze_me=True)
    dNoisy = mat['d']

    # Prepare data for analysis
    d = normalise_data(dNoisy)
    d = filter_data(d)

    # Find the peaks
    peaks = peakDetection(d, 2, 25)
    peaksDetected = len(peaks)

    # Scale data to prepare for input to NN
    d = scale_data(d)

    # Define neural network parameters
    input_nodes = 100
    hidden_nodes = 2*input_nodes
    output_nodes = 4
    learning_rate = 0.3

    # Create a NeuralNetwork instance (nn) with the desired parameters
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Set up array to hold class selections
    chosen_class = numpy.zeros(4)
    halfSampleSize = int(input_nodes/2)

    # Set up count and Class array
    count = 0
    Class = numpy.zeros(peaksDetected)

    # Check whether a network has been loaded
    if load_network != 0:
        nn.load(load_network)
    # Loop through all of the records in the data set
    for sample in peaks:
        inputs = numpy.asfarray(d[sample-halfSampleSize:sample+halfSampleSize])
        # Query the network
        outputs = nn.query(inputs)
        # The index of the highest value output corresponds to the label
        label = numpy.argmax(outputs) + 1
        print("Network label: ", label)
        # Log which neuron was chosen
        chosen_class[label-1] += 1
        Class[count] = label


    print("Number of peaks detected: ", peaksDetected)
    for neuron_type in range(4):
        print("Type ", neuron_type+1, ": ", int(chosen_class[neuron_type]))

    # Plot data and detected peaks if requested
    if plot_output == 1:
        plt.figure(1)
        plot_peaks(d, peaks)

    return d, peaks, Class

# Run the training program -- input args: input_file, load_network, plot_output (0 = No (Default), 1 = Yes)
#training_program('training.mat','saved.mat', 0, 0)
d, Index, Class = submission_program('submission.mat', 'saved.mat', 0)
save_submission('10236.mat', d, Index, Class)
