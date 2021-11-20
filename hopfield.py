import numpy as np
import random
import matplotlib.pyplot as plt


def recall(jij, iniState, t):
    E = 0
    s = iniState
    for k in range(t):
        ##update
        for x in range(len(s)):
            h = 0
            for y in range(len(s)):
                if x==y:
                    h += 0
                else:
                    h += jij[x,y]*s[y]
            if h > 0:
                s[x] = 1
            else:
                s[x] = -1
    return s

def training(mem):
    jij = np.zeros([mem.shape[1],mem.shape[1]])
    for k in range(mem.shape[0]):
        for i in range(mem.shape[1]):
            for j in range(i, mem.shape[1]):
                if i == j:
                    jij[i,j] = 0
                else:
                    jij[i,j] += mem[k,i]*mem[k,j]
                    jij[j,i] = jij[i,j]
    return jij/(mem.shape[0])

def ismember(training, recall):
    for i in range(training.shape[0]):
        result = 0
        for j in range(training.shape[1]):
            if recall[j] != training[i,j]:
                result += 1
        if result == 0:
            return f"Successful Recall of Memory {i}"
    return "Recall False Memory"

def ismemberb(training, recall):
    for i in range(training.shape[0]):
        result = 0
        for j in range(training.shape[1]):
            if recall[j] != training[i,j]:
                result += 1
        if result == 0:
            return 1
    return 0

def generateMem(size, number):
    mem = np.zeros([number, size])
    for i in range(number):
        for j in range(size):
            mem[i,j] = random.randrange(-1,2,2)
    return mem

def generateIni(size):
    mem = np.zeros(size)
    for j in range(size):
        mem[j] = random.randrange(-1,2,2)
    return mem



if __name__ == "__main__":
    ## n = 30, m = 1
    mem = np.array([[-1,1,1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,1]])
    jij = training(mem)
    iniState = np.array([1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState, 3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 30, m = 2
    mem = np.array([[1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],[1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    jij = training(mem)
    iniState = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState, 3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 30, m = 3
    mem = np.array([[-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
         [1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1],[1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
          -1, -1, -1],])
    jij = training(mem)
    iniState = np.array([1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))



    ## n = 30, m = 4
    mem = np.array([[1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1],
                    [1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1,
                     1],
                    [-1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1,
                     -1, 1],
                    [-1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1,
                     -1, -1],])

    jij = training(mem)
    iniState = np.array([-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 30, m = 5
    mem = np.array([[1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1],[-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1],[1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1]])
    jij = training(mem)
    iniState = np.array([1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1, -1,-1,-1,-1,1,1,1,1,-1,-1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 30, m = 6
    mem = np.array([[1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1],[-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1],[1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1],[-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1]])
    jij = training(mem)
    iniState = np.array([1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 30, m = 7
    mem = np.array([[1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1],[-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1],[1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1],[-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1],[1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1]])
    jij = training(mem)
    iniState = np.array([1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 30, m = 8
    mem = np.array([[1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,1,1,-1],[-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1],[1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1],[-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1],[1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1],[1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1]])
    jij = training(mem)
    iniState = np.array([1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 30, m = 9
    mem = np.array([[1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,1,1,-1],[-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1],[1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1],[-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1],[1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1],[1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1],[-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,1,1,1]])
    jij = training(mem)
    iniState = np.array([1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 30, m = 10
    mem = np.array([[1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1],[-1,1,1,-1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1],[-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1],[1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,-1],[-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1],[1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,-1],[1,-1,1,1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1],[-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,1,1,1],[1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1]])
    jij = training(mem)
    iniState = np.array([1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1])
    result = recall(jij, iniState,3)
    print(f"n=30, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 1
    mem = generateMem(50,1)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 2
    mem = generateMem(50,2)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 50, m = 3
    mem = generateMem(50,3)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 4
    mem = generateMem(50,4)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 5
    mem = generateMem(50,5)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 6
    mem = generateMem(50,6)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 7
    mem = generateMem(50,7)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 8
    mem = generateMem(50,8)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    ## n = 50, m = 9
    mem = generateMem(50,9)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    # n = 50, m = 10
    mem = generateMem(50,10)
    jij = training(mem)
    iniState = generateIni(50)
    result = recall(jij, iniState,3)
    print(f"n={mem.shape[1]}, m={mem.shape[0]}",result, ismember(mem,result))

    bar = np.zeros(100)
    n = 200
    for m in range(100):
        mem = generateMem(n,m+1)
        jij = training(mem)
        iniState = generateIni(n)
        result = recall(jij,iniState,3)
        bar[m] = ismemberb(mem, result)
        print(m)
    plt.plot(bar)
    plt.show()
