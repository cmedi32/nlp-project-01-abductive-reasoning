from datasets import load_dataset

train = load_dataset("amarfurt/art-nlp-hs26", split="train")
valid = load_dataset("amarfurt/art-nlp-hs26", split="validation")
test  = load_dataset("amarfurt/art-nlp-hs26", split="test")