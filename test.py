from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset

train_dataset = get_dataset("human_enhancers_ensembl", split="train")
test_dataset = get_dataset("human_enhancers_ensembl", split="test")

print(len(train_dataset))
seq, label = train_dataset[0]
print(seq[:50], label)