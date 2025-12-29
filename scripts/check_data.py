"""
Check dataset for issues
"""

import json
from collections import Counter

# Load data
with open('data/task_classifier_data.json', 'r') as f:
    data = json.load(f)

print("="*60)
print("DATASET DIAGNOSTIC")
print("="*60)

# Check splits
train_ids = set(item['text'] for item in data['train'])
val_ids = set(item['text'] for item in data['val'])
test_ids = set(item['text'] for item in data['test'])

overlap_train_val = train_ids & val_ids
overlap_train_test = train_ids & test_ids
overlap_val_test = val_ids & test_ids

print(f"\nSplit sizes:")
print(f"  Train: {len(data['train'])}")
print(f"  Val:   {len(data['val'])}")
print(f"  Test:  {len(data['test'])}")

print(f"\nOverlaps (should be 0):")
print(f"  Train ∩ Val:  {len(overlap_train_val)}")
print(f"  Train ∩ Test: {len(overlap_train_test)}")
print(f"  Val ∩ Test:   {len(overlap_val_test)}")

# Check label distribution
train_labels = Counter(item['label'] for item in data['train'])
print(f"\nTrain label distribution:")
for label, count in sorted(train_labels.items()):
    print(f"  {label:12s}: {count:6,d}")

# Check for label leakage
print(f"\nChecking for label leakage...")
leakage_count = 0
for item in data['train'][:100]:  # Check first 100
    text_lower = item['text'].lower()
    label = item['label']
    
    if label in text_lower:
        leakage_count += 1
        print(f"  LEAK: '{label}' found in text: {item['text'][:80]}...")

print(f"\nLabel leakage in first 100: {leakage_count}")

# Check synthetic vs real
train_synthetic = sum(1 for item in data['train'] if item.get('is_synthetic', False))
train_real = len(data['train']) - train_synthetic

print(f"\nTrain composition:")
print(f"  Real:      {train_real:6,d} ({train_real/len(data['train'])*100:.1f}%)")
print(f"  Synthetic: {train_synthetic:6,d} ({train_synthetic/len(data['train'])*100:.1f}%)")

# Check template patterns
print(f"\nChecking for template patterns...")
medical_starts = Counter()
code_starts = Counter()

for item in data['train']:
    if item['label'] == 'medical':
        first_words = ' '.join(item['text'].split()[:3])
        medical_starts[first_words] += 1
    elif item['label'] == 'code':
        first_words = ' '.join(item['text'].split()[:3])
        code_starts[first_words] += 1

print(f"\nTop medical start patterns:")
for pattern, count in medical_starts.most_common(5):
    print(f"  '{pattern}...': {count} times")

print(f"\nTop code start patterns:")
for pattern, count in code_starts.most_common(5):
    print(f"  '{pattern}...': {count} times")

print("="*60)