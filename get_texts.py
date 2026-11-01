import datasets
import h5py

dataset = datasets.load_dataset("croco-corp/pjm-segments", split="train")
with h5py.File("texts.h5", mode="w") as f:
    for record in dataset:
        f.create_dataset(record["__key__"], data=record["txt"])
    f.flush()
