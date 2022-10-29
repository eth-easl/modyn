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
    
    A_score = results.pop('A5', None)
    A_score = results.pop('A10', A_score)
    F_score = results.pop('F5', None)
    F_score = results.pop('F10', F_score)

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
        all_grad_err.extend(train_results.get('gradient_errors', [0 for _ in range(len(train_results['train_accuracy']))]))
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
    exp_name = 'blurry'
    results = {'0/1': {'train': {'train_loss': [0.6781425311691873, 0.04138552475439596], 'train_accuracy': [0.8109252745675113, 0.48603833177804895]}, 'val': {'0/1': {'Loss': 0.011630927454724238, 'Accuracy': 0.9947990543735225}, '2/3': {'Loss': 6.998963551236414, 'Accuracy': 0.0}, '4/5': {'Loss': 7.046033672487469, 'Accuracy': 0.0}, '6/7': {'Loss': 7.527572976138297, 'Accuracy': 0.0}, '8/9': {'Loss': 6.987492678686037, 'Accuracy': 0.0}, 'all': {'Loss': 5.6326707997878085, 'Accuracy': 0.2104}}}, '2/3': {'train': {'train_loss': [0.6049289267214101, 0.09653599470161249], 'train_accuracy': [0.8295137358601439, 0.4659697676167557]}, 'val': {'0/1': {'Loss': 0.0375426993564423, 'Accuracy': 0.991016548463357}, '2/3': {'Loss': 0.04317031059337526, 'Accuracy': 0.9872673849167483}, '4/5': {'Loss': 3.916842245877489, 'Accuracy': 0.0}, '6/7': {'Loss': 4.267029764188861, 'Accuracy': 0.0}, '8/9': {'Loss': 4.219516020980197, 'Accuracy': 0.0}, 'all': {'Loss': 2.4349340333427767, 'Accuracy': 0.4112}}}, '4/5': {'train': {'train_loss': [0.4758317580264296, 0.10126297585789464], 'train_accuracy': [0.8667367437514599, 0.4658164500766809]}, 'val': {'0/1': {'Loss': 0.07463235160947973, 'Accuracy': 0.9843971631205674}, '2/3': {'Loss': 0.07416750794964369, 'Accuracy': 0.9779627815866797}, '4/5': {'Loss': 0.030767395281359123, 'Accuracy': 0.9930629669156884}, '6/7': {'Loss': 2.9673600026367892, 'Accuracy': 0.07150050352467271}, '8/9': {'Loss': 3.313159629424533, 'Accuracy': 0.012607160867372668}, 'all': {'Loss': 1.2830128084030001, 'Accuracy': 0.6107}}}, '6/7': {'train': {'train_loss': [0.365846711803934, 0.12593168691417797], 'train_accuracy': [0.888290034897247, 0.45761510335049627]}, 'val': {'0/1': {'Loss': 0.09953464386636836, 'Accuracy': 0.9735224586288416}, '2/3': {'Loss': 0.16834722952616202, 'Accuracy': 0.9451518119490695}, '4/5': {'Loss': 0.1180724620564643, 'Accuracy': 0.959445037353255}, '6/7': {'Loss': 0.05969332467880914, 'Accuracy': 0.9848942598187311}, '8/9': {'Loss': 2.5089093371587032, 'Accuracy': 0.10640443772062531}, 'all': {'Loss': 0.586926676676143, 'Accuracy': 0.7954}}}, '8/9': {'train': {'train_loss': [0.41968300250117374, 0.07833005409854218], 'train_accuracy': [0.8701338084227285, 0.4753852080123267]}, 'val': {'0/1': {'Loss': 0.12029989792064853, 'Accuracy': 0.9716312056737588}, '2/3': {'Loss': 0.0702227631202524, 'Accuracy': 0.9769833496571988}, '4/5': {'Loss': 0.1104897008125403, 'Accuracy': 0.9631803628601922}, '6/7': {'Loss': 0.07690864211663019, 'Accuracy': 0.9763343403826787}, '8/9': {'Loss': 0.5511430645614216, 'Accuracy': 0.8411497730711044}, 'all': {'Loss': 0.18505441259853542, 'Accuracy': 0.9462}}}, 'A5': 0.9462, 'F5': 0.014378881490172347}
    visualize_results(results, exp_name)


    # with open(f'plots/{exp_name}/datadump') as f:
    #     results = f.read()
    #     print(results)