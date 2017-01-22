#Q-Learning parameters
REWARD = 5
GAMMA = 1
EXPLORE = 0.05
SAMPLE_BATCH = 500

#Sequence generation params
MAX_LEN = 10
MAX_STEPS = 20

#NN parameters
FC1 = 50
FC_OUT = 4

#Training params
LRATE = 1e-4
ADAM_EPS = 1e-16
N_EPOCH = 1
MINIBATCH = 50
PRINT_EPOCH = 1
SAVE_PATH = 'stack_model'

#Sequence match method
def seq_sim(str1,str2):
    eq = 0
    mn = min(len(str1),len(str2))
    mx = float(len(str1)+len(str2)-mn)
    if mx == 0:
        return 1
    for i in range(mn):
        if str1[i] == str2[i]:
            eq = eq+1
    return eq/mx
