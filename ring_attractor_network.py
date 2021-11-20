import numpy as np
import matplotlib.pyplot as plt

def local_net(stimulus):

    ##local jij
    jij = np.zeros([12,12])
    for i in range(12):
        for j in range(12):
            if i+1 == j or i-1 == j:
                jij[i,j] = 1
            else:
                jij[i,j] = -1
    jij[0,-1] = 1
    jij[-1,0] = 1


    tau = 0.2
    ring2 = np.zeros([12])
    for k in range(stimulus.shape[0]):
        ring = stimulus[k]
        for i in range(len(ring)):
            sum = 0
            for j in range(len(ring)):
                if j == i:
                    sum += 0
                else:
                    sum += jij[i,j]*ring[j]
            f = 1/2*(np.tanh(sum)+1)
            ring2[i] += -ring2[i]*tau + f + ring[i]
        plt.plot(ring2)
        plt.show()

    return ring2

def global_net(stimulus):
    deg = np.arange(12.) * 30.
    rad = np.radians(deg)

    ##global jij
    jij = np.zeros([12,12])
    for i in range(12):
        for j in range(12):
            jij[i,j] = np.cos(rad[i]-rad[j])

    tau = 0.2
    ring2 = np.zeros([12])
    for k in range(stimulus.shape[0]):
        ring = stimulus[k]
        for i in range(len(ring)):
            sum = 0
            for j in range(len(ring)):
                sum += jij[i,j]*ring[j]
            f = 1/2*(np.tanh(sum)+1)
            ring2[i] += -ring2[i]*tau + f + ring[i]
        plt.plot(ring2)
        plt.show()

    return ring2


if __name__ == "__main__":
    ## stimulus 1: 60 degree shift (wide), center 5 -> 3
    stimulus = np.array([[0, 0, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0, 0, 0],[0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0, 0, 0, 0, 0]])
    ## stimlus 2: 60 degree shift (narrow), center 5 -> 3
    # stimulus = np.array([[0, 0, 0, 0, 0.2, 1, 0.2, 0, 0, 0, 0, 0], [0, 0, 0.2, 1, 0.2, 0, 0, 0, 0, 0, 0, 0]])
    ## stimulus 3: 90 defree sift (wide), center 5 -> 2
    # stimulus = np.array([[0, 0, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0, 0, 0], [0.6, 0.8, 1, .8, .6, 0.4, 0, 0, 0, 0, 0, 0.4]])
    ## stimulus 4: 90 defree sift (narrow), center 5 -> 2
    # stimulus = np.array([[0, 0, 0, 0, 0.2, 1, 0.2, 0, 0, 0, 0, 0], [0, 0.2, 1, .2, 0, 0, 0, 0, 0, 0, 0, 0]])
    ## stimulus 5: 120 defree sift (wide), center 5 -> 1
    # stimulus = np.array([[0, 0, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0, 0, 0], [0.8, 1, .8, .6, .4, 0, 0, 0, 0, 0, 0.4, 0.6]])
    ## stimulus 6: 120 defree sift (narrow), center 5 -> 1
    # stimulus = np.array([[0, 0, 0, 0, 0.2, 1, 0.2, 0, 0, 0, 0, 0], [0.2, 1, .2, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    ## stimulus 7: 150 defree sift (wide), center 5 -> 0
    # stimulus = np.array([[0, 0, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0, 0, 0], [1, .8, .6, .4, 0, 0, 0, 0, 0, 0.4, 0.6, 0.8]])
    ## stimulus 8: 150 defree sift (narrow), center 5 -> 0
    # stimulus = np.array([[0, 0, 0, 0, 0.2, 1, 0.2, 0, 0, 0, 0, 0], [1, .2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2]])

    print(global_net(stimulus))
    print(local_net(stimulus))