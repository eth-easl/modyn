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

    grid_rows = len(results.keys()) // 3 + 2
    grid_cols = 3
    subplot_names = []
    extents = []
    num_tasks = len(results.keys())
    figsize=(14, 12)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    for idx, train_task in enumerate(results.keys()):
        validation_results = results[train_task]['val']

        labels = []
        y = []
        for k, v in validation_results.items():
            labels.append(k)
            y.append(v['Accuracy'])
        x = np.arange(len(labels))

        ax = fig.add_subplot(grid_rows, grid_cols, idx+1)
        ax.bar(x, y, align='center', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy')
        ax.set_title('Per task accuracy')
        subplot_names.append(train_task.replace('/', ' vs '))
        extents.append(ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()))

    tasks, all_loss, all_acc, all_grad_err = [], [], [], []
    for train_task in results.keys():
        train_results = results[train_task]['train']
        all_loss.extend(train_results['train_loss'])
        all_acc.extend(train_results['train_accuracy'])
        all_grad_err.extend(train_results['gradient_errors'])
        tasks.extend([train_task for _ in range(len(train_results['train_accuracy']))])

    epochs = len(all_loss) // len(results.keys())

    ax1 = fig.add_subplot(grid_rows, grid_cols, num_tasks+1)
    ax1.plot(all_loss)
    ax1.set_title('Train loss')
    ax1.set_xticks(range(len(all_loss)))
    ax1.set_xticklabels(tasks, rotation=20)
    [l.set_visible(False) for (i,l) in enumerate(ax1.xaxis.get_ticklabels()) if i % epochs != 0]
    subplot_names.append('train_loss')
    extents.append(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()))

    ax2 = fig.add_subplot(grid_rows, grid_cols, num_tasks+2)
    ax2.plot(all_acc)
    ax2.set_title('Train accuracy')
    ax2.set_xticks(range(len(all_acc)))
    ax2.set_xticklabels(tasks, rotation=20)
    [l.set_visible(False) for (i,l) in enumerate(ax2.xaxis.get_ticklabels()) if i % epochs != 0]
    subplot_names.append('train_accuracy')
    extents.append(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()))


    ax3 = fig.add_subplot(grid_rows, grid_cols, num_tasks+3)
    ax3.plot(all_grad_err)
    ax3.set_xticks(range(len(all_grad_err)))
    ax3.set_xticklabels(tasks, rotation=20)
    [l.set_visible(False) for (i,l) in enumerate(ax3.xaxis.get_ticklabels()) if i % epochs != 0]
    ax3.set_title('Gradient error')
    subplot_names.append('grad_error')
    extents.append(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()))

    plt.savefig(f'{dir_name}/all')

    for (name, extent) in zip(subplot_names, extents):
        fig.savefig(f'{dir_name}/{name}', bbox_inches=extent.expanded(1.4, 1.3))


if __name__ == '__main__':
    exp_name = 'pooling'
    results = {'0/1': {'train': {'train_loss': [0.271877747093777, 0.03378546297972293, 0.018586222202827043, 0.014533368378665859, 0.009747698910437315], 'train_accuracy': [0.8803000394788788, 0.9881563363600474, 0.9931306750888275, 0.9951835767864192, 0.9964469009080142], 'gradient_errors': [20.229305267333984, 9184.2431640625, 9376.873046875, 9583.134765625, 7941.06494140625]}, 'val': {'0/1': {'Loss': 0.0009916429687373948, 'Accuracy': 1.0}, '2/3': {'Loss': 9.137991127141191, 'Accuracy': 0.0}, '4/5': {'Loss': 9.766627632312103, 'Accuracy': 0.0}, '6/7': {'Loss': 10.742983100637568, 'Accuracy': 0.0}, '8/9': {'Loss': 10.786718938426423, 'Accuracy': 0.0}, 'all': {'Loss': 7.969016348221988, 'Accuracy': 0.2115}}}, '2/3': {'train': {'train_loss': [0.5552827379456858, 0.1281867124868044, 0.08914808864615947, 0.0748553942015216, 0.059367776786297656], 'train_accuracy': [0.8201502787428294, 0.9566938676577523, 0.9709945867334573, 0.9766906358568312, 0.9808111820311869], 'gradient_errors': [9911.283203125, 9023.306640625, 6680.11962890625, 7498.5283203125, 7830.88916015625]}, 'val': {'0/1': {'Loss': 0.15362071707739053, 'Accuracy': 0.9531914893617022}, '2/3': {'Loss': 0.01172943226354567, 'Accuracy': 0.9960822722820764}, '4/5': {'Loss': 14.258337919587387, 'Accuracy': 0.0}, '6/7': {'Loss': 11.859087205485395, 'Accuracy': 0.0}, '8/9': {'Loss': 12.479511039760819, 'Accuracy': 0.0}, 'all': {'Loss': 7.53680021605473, 'Accuracy': 0.405}}}, '4/5': {'train': {'train_loss': [0.4954903208351893, 0.14688093916390227, 0.1167025137427281, 0.10450868065902583, 0.0924753858580265], 'train_accuracy': [0.8505403346406704, 0.9510453993961796, 0.9609773990575887, 0.9662819897858412, 0.9698371942100956], 'gradient_errors': [8397.3154296875, 6696.109375, 6762.67724609375, 6576.029296875, 7479.9951171875]}, 'val': {'0/1': {'Loss': 0.17845651110013325, 'Accuracy': 0.9588652482269504}, '2/3': {'Loss': 0.2991393443186066, 'Accuracy': 0.9123408423114594}, '4/5': {'Loss': 0.005553395356851249, 'Accuracy': 0.9978655282817502}, '6/7': {'Loss': 17.272826848793606, 'Accuracy': 0.0}, '8/9': {'Loss': 15.514707487397764, 'Accuracy': 0.0}, 'all': {'Loss': 6.606818419418798, 'Accuracy': 0.5761}}}, '6/7': {'train': {'train_loss': [0.4848755797269004, 0.18082507244706475, 0.14087889441194604, 0.12582987304597013, 0.11101109385270005], 'train_accuracy': [0.863329852953972, 0.9409423127120856, 0.9542765161402593, 0.9593883233272427, 0.9637387975289307], 'gradient_errors': [6683.94482421875, 1656.44091796875, 2600.15576171875, 1846.666748046875, 2474.85205078125]}, 'val': {'0/1': {'Loss': 0.3060729449945139, 'Accuracy': 0.9215130023640662}, '2/3': {'Loss': 0.31274134681052024, 'Accuracy': 0.8996082272282077}, '4/5': {'Loss': 0.3441682644307677, 'Accuracy': 0.8959445037353255}, '6/7': {'Loss': 0.014505816449121406, 'Accuracy': 0.9959718026183283}, '8/9': {'Loss': 12.52281202863577, 'Accuracy': 0.0}, 'all': {'Loss': 2.679247824064642, 'Accuracy': 0.7443}}}, '8/9': {'train': {'train_loss': [0.4040111704212356, 0.18944619520578004, 0.15463239864092906, 0.13887081738702448, 0.12194572492221811], 'train_accuracy': [0.8857463820789003, 0.9405438445780744, 0.9513480473138174, 0.9568327496200356, 0.9615905636688032], 'gradient_errors': [2321.925537109375, 2363.8125, 2294.265625, 2629.8076171875, 2727.378662109375]}, 'val': {'0/1': {'Loss': 0.44608461918842146, 'Accuracy': 0.8732860520094563}, '2/3': {'Loss': 1.0857452818509064, 'Accuracy': 0.7801175318315378}, '4/5': {'Loss': 0.7405194746392004, 'Accuracy': 0.7977588046958378}, '6/7': {'Loss': 0.9274604189672019, 'Accuracy': 0.7739174219536757}, '8/9': {'Loss': 0.01602378663049463, 'Accuracy': 0.9949571356530509}, 'all': {'Loss': 0.6422005891554058, 'Accuracy': 0.8445}}}}
    visualize_results(results, exp_name)


    # with open(f'plots/{exp_name}/datadump') as f:
    #     results = f.read()
    #     print(results)