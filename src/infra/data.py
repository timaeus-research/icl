
import torch


class SeedableDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data: torch.utils.data.Dataset, *args, **kwargs):
        generator = torch.Generator(device="cpu")
        sampler = torch.utils.data.RandomSampler(data, generator=generator)
        super().__init__(data, *args, sampler=sampler, **kwargs)

    def set_seed(self, seed: int):
        self.sampler.generator.manual_seed(seed)
