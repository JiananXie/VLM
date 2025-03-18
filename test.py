import numpy as np

len_dict = {'sys': 1, 'img': 10, 'inst': 10, 'out': 10}

print(np.array(range(len_dict['img']))/np.array(range(len_dict['img'], 0, -1)))

