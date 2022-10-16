import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_results(results, exp_name):
    dir_name = f'plots/{exp_name}'
     
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f = open(dir_name+'/datadump', "w")
    f.write(str(results))
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
        plt.savefig('{}/{}'.format(dir_name, train_task.replace('/', ' vs ')))
        plt.close()

if __name__ == '__main__':
    exp_name = 'cifar2'
    results = {'Airplane/Car': {'Airplane/Car': {'Loss': 0.4217320756912231, 'Accuracy': 0.8025}, 'Bird/Cat': {'Loss': 0.7344066467285156, 'Accuracy': 0.634}, 'Deer/Dog': {'Loss': 0.8213122382164001, 'Accuracy': 0.5665}, 'Frog/Horse': {'Loss': 0.8923384084701538, 'Accuracy': 0.5005}, 'Ship/Truck': {'Loss': 0.642521698474884, 'Accuracy': 0.687}, 'all': {'Loss': 0.7024622135162354, 'Accuracy': 0.6381}}, 'Bird/Cat': {'Airplane/Car': {'Loss': 0.6552122592926025, 'Accuracy': 0.658}, 'Bird/Cat': {'Loss': 0.5872413151264191, 'Accuracy': 0.703}, 'Deer/Dog': {'Loss': 0.6196407027244568, 'Accuracy': 0.6815}, 'Frog/Horse': {'Loss': 0.7141637086868287, 'Accuracy': 0.6}, 'Ship/Truck': {'Loss': 0.7260330986976623, 'Accuracy': 0.592}, 'all': {'Loss': 0.6604582169055939, 'Accuracy': 0.6469}}, 'Deer/Dog': {'Airplane/Car': {'Loss': 0.5741859457492828, 'Accuracy': 0.6975}, 'Bird/Cat': {'Loss': 0.6489504036903382, 'Accuracy': 0.6695}, 'Deer/Dog': {'Loss': 0.48334412622451783, 'Accuracy': 0.7665}, 'Frog/Horse': {'Loss': 0.8661722574234009, 'Accuracy': 0.505}, 'Ship/Truck': {'Loss': 0.5349745013713837, 'Accuracy': 0.7495}, 'all': {'Loss': 0.6215254468917847, 'Accuracy': 0.6776}}, 'Frog/Horse': {'Airplane/Car': {'Loss': 1.2315180530548095, 'Accuracy': 0.527}, 'Bird/Cat': {'Loss': 0.944965886592865, 'Accuracy': 0.5835}, 'Deer/Dog': {'Loss': 0.9760890474319458, 'Accuracy': 0.5845}, 'Frog/Horse': {'Loss': 0.35511610740423205, 'Accuracy': 0.8435}, 'Ship/Truck': {'Loss': 1.3233615503311158, 'Accuracy': 0.5195}, 'all': {'Loss': 0.9662101289629936, 'Accuracy': 0.6116}}, 'Ship/Truck': {'Airplane/Car': {'Loss': 0.6498887796401978, 'Accuracy': 0.72}, 'Bird/Cat': {'Loss': 1.1443447756767273, 'Accuracy': 0.561}, 'Deer/Dog': {'Loss': 1.2160776300430298, 'Accuracy': 0.5325}, 'Frog/Horse': {'Loss': 1.4537734112739562, 'Accuracy': 0.517}, 'Ship/Truck': {'Loss': 0.517494707107544, 'Accuracy': 0.7635}, 'all': {'Loss': 0.996315860748291, 'Accuracy': 0.6188}}}
    visualize_results(results, exp_name)


    # with open(f'plots/{exp_name}/datadump') as f:
    #     results = f.read()
    #     print(results)