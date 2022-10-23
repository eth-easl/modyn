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
    exp_name = 'cifar3'
    # results = {'Airplane/Car': {'Airplane/Car': {'Loss': 0.4217320756912231, 'Accuracy': 0.8025}, 'Bird/Cat': {'Loss': 0.7344066467285156, 'Accuracy': 0.634}, 'Deer/Dog': {'Loss': 0.8213122382164001, 'Accuracy': 0.5665}, 'Frog/Horse': {'Loss': 0.8923384084701538, 'Accuracy': 0.5005}, 'Ship/Truck': {'Loss': 0.642521698474884, 'Accuracy': 0.687}, 'all': {'Loss': 0.7024622135162354, 'Accuracy': 0.6381}}, 'Bird/Cat': {'Airplane/Car': {'Loss': 0.6552122592926025, 'Accuracy': 0.658}, 'Bird/Cat': {'Loss': 0.5872413151264191, 'Accuracy': 0.703}, 'Deer/Dog': {'Loss': 0.6196407027244568, 'Accuracy': 0.6815}, 'Frog/Horse': {'Loss': 0.7141637086868287, 'Accuracy': 0.6}, 'Ship/Truck': {'Loss': 0.7260330986976623, 'Accuracy': 0.592}, 'all': {'Loss': 0.6604582169055939, 'Accuracy': 0.6469}}, 'Deer/Dog': {'Airplane/Car': {'Loss': 0.5741859457492828, 'Accuracy': 0.6975}, 'Bird/Cat': {'Loss': 0.6489504036903382, 'Accuracy': 0.6695}, 'Deer/Dog': {'Loss': 0.48334412622451783, 'Accuracy': 0.7665}, 'Frog/Horse': {'Loss': 0.8661722574234009, 'Accuracy': 0.505}, 'Ship/Truck': {'Loss': 0.5349745013713837, 'Accuracy': 0.7495}, 'all': {'Loss': 0.6215254468917847, 'Accuracy': 0.6776}}, 'Frog/Horse': {'Airplane/Car': {'Loss': 1.2315180530548095, 'Accuracy': 0.527}, 'Bird/Cat': {'Loss': 0.944965886592865, 'Accuracy': 0.5835}, 'Deer/Dog': {'Loss': 0.9760890474319458, 'Accuracy': 0.5845}, 'Frog/Horse': {'Loss': 0.35511610740423205, 'Accuracy': 0.8435}, 'Ship/Truck': {'Loss': 1.3233615503311158, 'Accuracy': 0.5195}, 'all': {'Loss': 0.9662101289629936, 'Accuracy': 0.6116}}, 'Ship/Truck': {'Airplane/Car': {'Loss': 0.6498887796401978, 'Accuracy': 0.72}, 'Bird/Cat': {'Loss': 1.1443447756767273, 'Accuracy': 0.561}, 'Deer/Dog': {'Loss': 1.2160776300430298, 'Accuracy': 0.5325}, 'Frog/Horse': {'Loss': 1.4537734112739562, 'Accuracy': 0.517}, 'Ship/Truck': {'Loss': 0.517494707107544, 'Accuracy': 0.7635}, 'all': {'Loss': 0.996315860748291, 'Accuracy': 0.6188}}}
    results = {'Airplane/Car': {'train': {'train_loss': [0.6174429515838623, 0.4645849326133728], 'train_accuracy': [0.7134, 0.785], 'gradient_errors': [105.94542694091797, 31242.94140625]}, 'val': {'Airplane/Car': {'Loss': 0.5271902017593384, 'Accuracy': 0.727}, 'Bird/Cat': {'Loss': 8.454095466613769, 'Accuracy': 0.0}, 'Deer/Dog': {'Loss': 8.547009422302246, 'Accuracy': 0.0}, 'Frog/Horse': {'Loss': 8.5438934173584, 'Accuracy': 0.0}, 'Ship/Truck': {'Loss': 8.308535102844239, 'Accuracy': 0.0}, 'all': {'Loss': 6.8761447221755985, 'Accuracy': 0.1454}}}, 'Bird/Cat': {'train': {'train_loss': [1.0221489809513091, 0.6849585904121399], 'train_accuracy': [0.5292, 0.682], 'gradient_errors': [32373.013671875, 120413.671875]}, 'val': {'Airplane/Car': {'Loss': 4.140077291488647, 'Accuracy': 0.0775}, 'Bird/Cat': {'Loss': 1.0413516602516175, 'Accuracy': 0.5555}, 'Deer/Dog': {'Loss': 10.997130043029784, 'Accuracy': 0.0}, 'Frog/Horse': {'Loss': 10.917494384765625, 'Accuracy': 0.0}, 'Ship/Truck': {'Loss': 10.592695114135742, 'Accuracy': 0.0}, 'all': {'Loss': 7.537749698734284, 'Accuracy': 0.1266}}}, 'Deer/Dog': {'train': {'train_loss': [1.081700277376175, 0.5997486578941346], 'train_accuracy': [0.6395000000000001, 0.7652], 'gradient_errors': [61768.8125, 114466.734375]}, 'val': {'Airplane/Car': {'Loss': 4.224767702102661, 'Accuracy': 0.1165}, 'Bird/Cat': {'Loss': 6.692882907867432, 'Accuracy': 0.077}, 'Deer/Dog': {'Loss': 1.2844171657562256, 'Accuracy': 0.5995}, 'Frog/Horse': {'Loss': 12.525343139648438, 'Accuracy': 0.0}, 'Ship/Truck': {'Loss': 13.269469123840333, 'Accuracy': 0.0}, 'all': {'Loss': 7.599376007843017, 'Accuracy': 0.1586}}}, 'Frog/Horse': {'train': {'train_loss': [1.3201434776306151, 0.6856105953216552], 'train_accuracy': [0.6017, 0.7127], 'gradient_errors': [137256.046875, 42069.65625]}, 'val': {'Airplane/Car': {'Loss': 9.860193359375, 'Accuracy': 0.0}, 'Bird/Cat': {'Loss': 12.655618003845214, 'Accuracy': 0.008}, 'Deer/Dog': {'Loss': 8.486122932434082, 'Accuracy': 0.0005}, 'Frog/Horse': {'Loss': 0.7701689410209656, 'Accuracy': 0.641}, 'Ship/Truck': {'Loss': 19.418195449829103, 'Accuracy': 0.0}, 'all': {'Loss': 10.238059737300873, 'Accuracy': 0.1299}}}, 'Ship/Truck': {'train': {'train_loss': [1.3507703936576843, 0.7478316677093506], 'train_accuracy': [0.5989, 0.7014], 'gradient_errors': [61137.03515625, 18401.689453125]}, 'val': {'Airplane/Car': {'Loss': 8.401227172851563, 'Accuracy': 0.0}, 'Bird/Cat': {'Loss': 10.082863891601562, 'Accuracy': 0.0}, 'Deer/Dog': {'Loss': 6.657298046112061, 'Accuracy': 0.004}, 'Frog/Horse': {'Loss': 10.700707683563232, 'Accuracy': 0.0}, 'Ship/Truck': {'Loss': 0.6673767018318176, 'Accuracy': 0.6825}, 'all': {'Loss': 7.301894699192047, 'Accuracy': 0.1373}}}}
    visualize_results(results, exp_name)


    # with open(f'plots/{exp_name}/datadump') as f:
    #     results = f.read()
    #     print(results)