from torch.utils.data import DataLoader, Dataset


class TaskDataset(Dataset):
    def __init__(self, tasks, dataset_configs):
        self.tasks = tasks
        self.active_idx = 0
        self.blurry = dataset_configs.get('blurry', 0)

    def set_active_task(self, task):
        if isinstance(task, str):
            task = self.tasks.index(task)
        self.active_idx = task
    
    def active_task(self):
        return self.tasks[self.active_idx]

    def is_task_based(self):
        return True

    def has_more_tasks(self):
        return self.active_idx < len(self.tasks) - 1
    
    def next_task(self):
        if self.has_more_tasks():
            self.active_idx += 1
        else:
            raise IndexError()

