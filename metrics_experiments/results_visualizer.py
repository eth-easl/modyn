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
        validation_results = results[train_task]['val']

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

    tasks, all_loss, all_acc, all_grad_err = [], [], [], []
    for train_task in results.keys():
        train_results = results[train_task]['train']
        all_loss.extend(train_results['train_loss'])
        all_acc.extend(train_results['train_accuracy'])
        all_grad_err.extend(train_results['gradient_errors'])
        tasks.extend([train_task for _ in range(len(train_results['train_accuracy']))])

    epochs = len(all_loss) // len(results.keys())

    plt.figure()
    plt.plot(all_loss)
    plt.title('Train loss')
    plt.xticks(range(len(all_loss)), tasks, rotation=20)
    [l.set_visible(False) for (i,l) in enumerate(plt.gca().xaxis.get_ticklabels()) if i % epochs != 0]
    plt.savefig(f'{dir_name}/train_loss')
    plt.close()

    plt.figure()
    plt.plot(all_acc)
    plt.title('Train accuracy')
    plt.xticks(range(len(all_acc)), tasks, rotation=20)
    [l.set_visible(False) for (i,l) in enumerate(plt.gca().xaxis.get_ticklabels()) if i % epochs != 0]
    plt.savefig(f'{dir_name}/train_accuracy')
    plt.close()

    plt.figure()
    plt.plot(all_grad_err)
    plt.xticks(range(len(all_grad_err)), tasks, rotation=20)
    [l.set_visible(False) for (i,l) in enumerate(plt.gca().xaxis.get_ticklabels()) if i % epochs != 0]
    plt.title('Gradient error')
    plt.savefig(f'{dir_name}/grad_error')
    plt.close()



if __name__ == '__main__':
    exp_name = 'exp3'
    # results = {'Airplane/Car': {'Airplane/Car': {'Loss': 0.4217320756912231, 'Accuracy': 0.8025}, 'Bird/Cat': {'Loss': 0.7344066467285156, 'Accuracy': 0.634}, 'Deer/Dog': {'Loss': 0.8213122382164001, 'Accuracy': 0.5665}, 'Frog/Horse': {'Loss': 0.8923384084701538, 'Accuracy': 0.5005}, 'Ship/Truck': {'Loss': 0.642521698474884, 'Accuracy': 0.687}, 'all': {'Loss': 0.7024622135162354, 'Accuracy': 0.6381}}, 'Bird/Cat': {'Airplane/Car': {'Loss': 0.6552122592926025, 'Accuracy': 0.658}, 'Bird/Cat': {'Loss': 0.5872413151264191, 'Accuracy': 0.703}, 'Deer/Dog': {'Loss': 0.6196407027244568, 'Accuracy': 0.6815}, 'Frog/Horse': {'Loss': 0.7141637086868287, 'Accuracy': 0.6}, 'Ship/Truck': {'Loss': 0.7260330986976623, 'Accuracy': 0.592}, 'all': {'Loss': 0.6604582169055939, 'Accuracy': 0.6469}}, 'Deer/Dog': {'Airplane/Car': {'Loss': 0.5741859457492828, 'Accuracy': 0.6975}, 'Bird/Cat': {'Loss': 0.6489504036903382, 'Accuracy': 0.6695}, 'Deer/Dog': {'Loss': 0.48334412622451783, 'Accuracy': 0.7665}, 'Frog/Horse': {'Loss': 0.8661722574234009, 'Accuracy': 0.505}, 'Ship/Truck': {'Loss': 0.5349745013713837, 'Accuracy': 0.7495}, 'all': {'Loss': 0.6215254468917847, 'Accuracy': 0.6776}}, 'Frog/Horse': {'Airplane/Car': {'Loss': 1.2315180530548095, 'Accuracy': 0.527}, 'Bird/Cat': {'Loss': 0.944965886592865, 'Accuracy': 0.5835}, 'Deer/Dog': {'Loss': 0.9760890474319458, 'Accuracy': 0.5845}, 'Frog/Horse': {'Loss': 0.35511610740423205, 'Accuracy': 0.8435}, 'Ship/Truck': {'Loss': 1.3233615503311158, 'Accuracy': 0.5195}, 'all': {'Loss': 0.9662101289629936, 'Accuracy': 0.6116}}, 'Ship/Truck': {'Airplane/Car': {'Loss': 0.6498887796401978, 'Accuracy': 0.72}, 'Bird/Cat': {'Loss': 1.1443447756767273, 'Accuracy': 0.561}, 'Deer/Dog': {'Loss': 1.2160776300430298, 'Accuracy': 0.5325}, 'Frog/Horse': {'Loss': 1.4537734112739562, 'Accuracy': 0.517}, 'Ship/Truck': {'Loss': 0.517494707107544, 'Accuracy': 0.7635}, 'all': {'Loss': 0.996315860748291, 'Accuracy': 0.6188}}}
    results = {'0/1': {'train': {'train_loss': [0.34143966927424374, 0.08538813336262979, 0.04472544072787846, 0.031997827686695915], 'train_accuracy': [0.905250690880379, 0.9692854322937229, 0.9867350967232531, 0.9904461113304383], 'gradient_errors': [68.3343734741211, 3416.440185546875, 3585.347900390625, 3703.8798828125]}, 'val': {'0/1': {'Loss': 0.0033271383987321357, 'Accuracy': 0.9990543735224586}, '2/3': {'Loss': 34.43867017426523, 'Accuracy': 0.0}, '4/5': {'Loss': 29.569594823945167, 'Accuracy': 0.0}, '6/7': {'Loss': 31.358858926778833, 'Accuracy': 0.0}, '8/9': {'Loss': 32.76534647323601, 'Accuracy': 0.0}, 'all': {'Loss': 25.299659797864592, 'Accuracy': 0.2113}}}, '2/3': {'train': {'train_loss': [0.6458926307365196, 0.39249601861325945, 0.31054367316830234, 0.27697245853075825], 'train_accuracy': [0.8235586069980975, 0.8963520555877245, 0.9372983704193896, 0.9513607411696584], 'gradient_errors': [67.56340026855469, 2602.22705078125, 2485.91064453125, 2519.46240234375]}, 'val': {'0/1': {'Loss': 0.8341329770167105, 'Accuracy': 0.7021276595744681}, '2/3': {'Loss': 0.03020970768547899, 'Accuracy': 0.9906953966699314}, '4/5': {'Loss': 22.92770728739247, 'Accuracy': 0.0}, '6/7': {'Loss': 24.392129519674832, 'Accuracy': 0.0}, '8/9': {'Loss': 25.142018484095402, 'Accuracy': 0.0}, 'all': {'Loss': 14.309179480609298, 'Accuracy': 0.3508}}}, '4/5': {'train': {'train_loss': [0.7248263787039959, 0.39165449390389784, 0.2832591406968605, 0.24515641762818047], 'train_accuracy': [0.8304181834324781, 0.9251531563526592, 0.9627985439048211, 0.9766492053626921], 'gradient_errors': [68.27378845214844, 2284.072998046875, 1895.2547607421875, 1771.6436767578125]}, 'val': {'0/1': {'Loss': 0.8925260984587613, 'Accuracy': 0.7531914893617021}, '2/3': {'Loss': 1.2756720023570869, 'Accuracy': 0.5053868756121449}, '4/5': {'Loss': 0.010910964782648941, 'Accuracy': 0.9994663820704376}, '6/7': {'Loss': 17.47036670942201, 'Accuracy': 0.0}, '8/9': {'Loss': 20.95306805655383, 'Accuracy': 0.0}, 'all': {'Loss': 8.075914431611448, 'Accuracy': 0.4498}}}, '6/7': {'train': {'train_loss': [0.6067010033436913, 0.2862515717385233, 0.2251154265164605, 0.19512141950512268], 'train_accuracy': [0.9038824591644093, 0.9725847492407452, 0.9880160879914635, 0.991873922679143], 'gradient_errors': [64.59259033203125, 1935.6292724609375, 1550.72509765625, 1384.7813720703125]}, 'val': {'0/1': {'Loss': 1.5533804005201264, 'Accuracy': 0.41843971631205673}, '2/3': {'Loss': 1.7664351381587702, 'Accuracy': 0.4520078354554358}, '4/5': {'Loss': 3.670479180591465, 'Accuracy': 0.004268943436499467}, '6/7': {'Loss': 0.017264063100537803, 'Accuracy': 0.9974823766364552}, '8/9': {'Loss': 15.42972446726358, 'Accuracy': 0.0}, 'all': {'Loss': 4.4402368131550025, 'Accuracy': 0.3797}}}, '8/9': {'train': {'train_loss': [0.811812328362869, 0.500349358421261, 0.40987586075976745, 0.35449096162440413], 'train_accuracy': [0.7947457627118645, 0.8864406779661017, 0.9296610169491526, 0.9467796610169492], 'gradient_errors': [68.26251220703125, 1697.939453125, 1425.2559814453125, 1281.3802490234375]}, 'val': {'0/1': {'Loss': 2.8994134931135966, 'Accuracy': 0.09787234042553192}, '2/3': {'Loss': 3.7393821390321977, 'Accuracy': 0.12193927522037218}, '4/5': {'Loss': 5.434557727459528, 'Accuracy': 0.0005336179295624333}, '6/7': {'Loss': 4.457522308718402, 'Accuracy': 0.0}, '8/9': {'Loss': 0.03511607725656039, 'Accuracy': 0.9909228441754917}, 'all': {'Loss': 3.2874713533412665, 'Accuracy': 0.2422}}}}
    visualize_results(results, exp_name)


    # with open(f'plots/{exp_name}/datadump') as f:
    #     results = f.read()
    #     print(results)