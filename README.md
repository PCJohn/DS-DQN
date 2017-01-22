# DS-DQN

Deep Q-Networks for training a model to learn how to use basic data structures.
This is still in progress, so quite a lot needs to be fixed and cleaned up

I'm trying to use Q Learning to train a model to use data structures. For example, to train a model to use a stack to reverse a string, we simply give it lots of input-output sequence pairs. The output on a new sequence will be the sequence of operations the model does on the stack to get the desired output sequence.
Note that this in fact, is exactly what neural stacks do. However, I'm trying to treat the data structure as a black box (not change its architecture) so I can easily add a new data structure and interafce it with my model.

Not sure if this will work!
