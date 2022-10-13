import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_results(results, exp_name):
    dir_name = f'plots/{exp_name}'
     
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f = open(dir_name+'/datadump', "w")
    f.write(results)
    f.close()

    for train_task in results.keys():
        validation_results = results[train_task]

        labels = []
        y = []
        for k, v in validation_results.items():
            labels.append(k)
            y.append(v['Accuracy'])
        x = np.arange(len(labels))

        plt.figure()
        plt.bar(x, y, align='center', alpha=0.5)
        plt.xticks(x, labels)
        plt.ylabel('Accuracy')
        plt.title('Per task accuracy')
        plt.savefig('{}/{}'.format(dir_name, train_task[0]))
        plt.close()
