#!/bin/bash

# Download datasets for MemCtrl training
# Run from project root: bash scripts/download_datasets.sh

set -e  # Exit on error

echo " Downloading datasets for MemCtrl..."

# Create data directory
mkdir -p data/datasets
cd data/datasets

# ============================================
# 1. LongBench (Primary Evaluation)
# ============================================
echo ""
echo "  Downloading LongBench..."
if [ ! -d "LongBench" ]; then
    git clone https://github.com/THUDM/LongBench.git
    cd LongBench
    # Download data
    wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
    unzip -q data.zip
    rm data.zip
    cd ..
    echo "✓ LongBench downloaded"
else
    echo "✓ LongBench already exists"
fi

# ============================================
# 2. MedDialog (Medical conversations)
# ============================================
echo ""
echo "  Downloading MedDialog..."
if [ ! -d "meddialog" ]; then
    mkdir -p meddialog
    cd meddialog
    # Download from Hugging Face
    wget https://huggingface.co/datasets/medical_dialog/resolve/main/en/train.txt
    wget https://huggingface.co/datasets/medical_dialog/resolve/main/en/val.txt
    cd ..
    echo "✓ MedDialog downloaded"
else
    echo "✓ MedDialog already exists"
fi

# ============================================
# 3. Ubuntu Dialogue (Code/Tech support)
# ============================================
echo ""
echo "  Downloading Ubuntu Dialogue Corpus..."
if [ ! -d "ubuntu" ]; then
    mkdir -p ubuntu
    cd ubuntu
    # This is a large dataset, download sample
    wget http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz
    tar -xzf ubuntu_dialogs.tgz
    rm ubuntu_dialogs.tgz
    cd ..
    echo "✓ Ubuntu Dialogue downloaded"
else
    echo "✓ Ubuntu Dialogue already exists"
fi

# ============================================
# 4. DailyDialog (General conversation)
# ============================================
echo ""
echo " Downloading DailyDialog..."
if [ ! -d "dailydialog" ]; then
    mkdir -p dailydialog
    cd dailydialog
    wget http://yanran.li/files/ijcnlp_dailydialog.zip
    unzip -q ijcnlp_dailydialog.zip
    rm ijcnlp_dailydialog.zip
    cd ..
    echo "✓ DailyDialog downloaded"
else
    echo "✓ DailyDialog already exists"
fi

# ============================================
# 5. CodeSearchNet (Code conversations)
# ============================================
echo ""
echo "   Setting up CodeSearchNet (using Hugging Face)..."
echo "   (Will be downloaded via Python script)"

# ============================================
# 6. WritingPrompts (Creative writing)
# ============================================
echo ""
echo "   Setting up WritingPrompts..."
echo "   (Will be downloaded via Hugging Face datasets)"

cd ../..  # Back to project root

echo ""
echo " Dataset download complete!"
echo ""
echo "Downloaded datasets:"
echo "  - LongBench (evaluation)"
echo "  - MedDialog (medical training)"
echo "  - Ubuntu Dialogue (code training)"
echo "  - DailyDialog (general training)"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/prepare_task_classifier_data.py"
echo "  2. Run: python scripts/train_task_classifier.py"