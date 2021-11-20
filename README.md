# Attractor_Networks

## Drosophila Ring Attractor Network: ring_attractor_network.py

Implemented ring attractor networks of Drosophila’s head direction system described in Kim et al. (2017) in Python and explored the difference between global network model and local network model. 

The model takes two inputs, a 12 elements array of visual input of initial head direction and that of shifted head direction, and outputs the strength of visual input presented in each neuron in the ring attractor after the shift.

Kim SS, Rouault H, Druckmann S, Jayaraman V. Ring attractor dynamics in the Drosophila central brain. Science. 2017 May 26;356(6340):849-853. doi: 10.1126/science.aal4835. Epub 2017 May 4. PMID: 28473639. 

## Hopfield Network of Long-term Associative Memory: hopfield.py

Implemented a discrete state Hopfield network in Python and explored the relationship between initial state, recall precision, number of stored memories, and number of neurons. 

The model has two functions, the first training function takes one input, memories (an array consists of arrays of 1 and -1), and trains them into a memory matrix. The second recall function takes three inputs: the memory matrix, initial state (an array consisting of 1 and -1), and simulation time. The recall function outputs the recalled (an array consisting of 1 and -1) memory from the matrix based on the initial state and simulation time.
