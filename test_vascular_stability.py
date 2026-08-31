import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig
from train_downstream import build_multilabel_patient_sampler, train_epoch


class _PatientFiles:
    def __init__(self, root, labels):
        self.data_dir = str(root)
        self.disease_labels = ["冠心病", "脑卒中（中风）"]
        self.files = []
        for index, values in enumerate(labels):
            filename = f"train_uid{index}_0.pkl"
            with open(root / filename, "wb") as handle:
                pickle.dump({
                    "label": dict(zip(self.disease_labels, values)),
                    "data": [0.0],
                }, handle)
            self.files.append(filename)

    def __len__(self):
        return len(self.files)


def test_balanced_patient_sampler_supports_stroke(tmp_path):
    dataset = _PatientFiles(tmp_path, [(0, 0), (1, 0), (0, 1), (0, 0)])
    config = TrainConfig()
    config.multidisease_sampler_mode = "multilabel_balanced"
    sampler = build_multilabel_patient_sampler(dataset, config)
    assert len(sampler) == len(dataset)
    assert sampler.replacement is True


def test_gradient_accumulation_controls_optimizer_steps():
    x = torch.tensor([[0.01], [0.02], [0.03], [0.04]])
    y = torch.tensor([0, 1, 0, 1])
    uids = ["a", "b", "c", "d"]

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return x[index], y[index], uids[index]

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=0.01)
            self.step_count = 0

        def step(self, closure=None):
            self.step_count += 1
            return super().step(closure)

    class Scheduler:
        def __init__(self):
            self.step_count = 0

        def step(self):
            self.step_count += 1

    model = torch.nn.Linear(1, 2)
    optimizer = CountingSGD(model.parameters())
    scheduler = Scheduler()
    train_epoch(
        model,
        DataLoader(Dataset(), batch_size=1, shuffle=False),
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        scheduler=scheduler,
        sched_mode="batch",
        gradient_accumulation_steps=2,
    )
    assert optimizer.step_count == 2
    assert scheduler.step_count == 2
