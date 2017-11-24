import copy
import numpy as np

class Buffer(object):
    def __init__(self, nbucket, loss_name_dict, batch_size):
        self.nbucket = nbucket
        self.batch_size = batch_size
        self.data = {}
        for item in loss_name_dict:
            self.data[item] = [[],[],[],[]]

    def add(self, loss_name, x, x_mask, y, y_mask):
        self.data[loss_name][0].append(x)
        self.data[loss_name][1].append(x_mask)
        self.data[loss_name][2].append(y)
        self.data[loss_name][3].append(y_mask)

    def check(self):
        data_for_train = {}
        for key in self.data:
            if(len(self.data[key][0])>self.batch_size):
                data_for_train[key] = [copy.deepcopy(self.data[key][0][:self.batch_size]),
                                       copy.deepcopy(self.data[key][1][:self.batch_size]),
                                       copy.deepcopy(self.data[key][2][:self.batch_size]),
                                       copy.deepcopy(self.data[key][3][:self.batch_size])
                                       ]
                self.data[key] = [self.data[key][0][self.batch_size:],
                                  self.data[key][1][self.batch_size:],
                                  self.data[key][2][self.batch_size:],
                                  self.data[key][3][self.batch_size:]
                                  ]
        return data_for_train

# test for buffer
'''
loss_name_dict = {'CE','M','F','B'}
buffer = Buffer(len(loss_name_dict),loss_name_dict,8)
while(True):
    for i in range(0,8):
        temp = np.random.randint(0,4)
        if temp == 0:
            name = 'CE'
        elif temp == 1:
            name = 'M'
        elif temp == 2:
            name = 'F'
        else:
            name = 'B'

        buffer.add(name,1,1,1,1)

    return_value = buffer.check()
    a = 1
'''