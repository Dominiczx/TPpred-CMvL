import matplotlib.pyplot as plt
import re

pattern = re.compile('epoch:(.*?) tag: (.*?) acc: (.*?)$')
val, test = [0] * 93, [0] * 100
with open('/home/czx/workspace/TPpred-Cons/save_model/fl4_lr_0.0005_pssmweight_0.17.txt', "r", encoding='utf-8') as f:
    for line in f:
        res = pattern.findall(line)
        epoch, tag, acc = int(res[0][0]), res[0][1], float(res[0][2])
        if tag == 'val':
            tmp_epoch = epoch
            val[tmp_epoch - 1] = acc
        else:
            test[tmp_epoch - 1] = acc
 
fig = plt.figure()
val_x, test_x = [i+1 for i in range(len(val))], [i+1 for i in range(len(test)) if test[i] != 0]
test = [i for i in test if i != 0]
val_line = plt.plot(val_x, val, color='r', linestyle='-', label = 'val_line')
test_line = plt.plot(test_x, test, color="b", linestyle='solid', label='test_line')
plt.text(val_x[val.index(max(val))], max(val), round(max(val),3), horizontalalignment='center')
plt.legend()
plt.savefig("plot/fl4.png",dpi=300)
        