# Continual Learning Multi-Task Model

A PyTorch implementation of a continual learning system that sequentially learns three computer vision tasks: semantic segmentation, object detection, and image classification, while mitigating catastrophic forgetting using Elastic Weight Consolidation (EWC) and other regularization techniques.

## Overview

This project implements a multi-task learning framework that can:
- Learn multiple vision tasks sequentially without forgetting previous tasks
- Handle VOC segmentation, COCO detection, and ImageNet classification
- Use advanced regularization techniques to prevent catastrophic forgetting
- Provide comprehensive evaluation and visualization tools

## Architecture

### Core Components

#### 1. BalancedMultiTaskModel
A unified neural network architecture based on **pretrained EfficientNet-B0** that handles three different tasks:
- **Shared Backbone**: EfficientNet-B0 feature extractor (pretrained on ImageNet)
- **Task-specific Heads**:
  - **Segmentation branch**: Upsampling layers for pixel-level predictions → `[batch, 1, 224, 224]`
  - **Detection branch**: Multi-scale feature fusion → `[batch, 80]` (COCO classes)
  - **Classification branch**: Enhanced capacity → `[batch, 30]` (ImageNet subset)

**Model Specifications:**
- **Total Parameters**: 6.16M (within 8M limit)
- **Input Shape**: `[batch, 3, 224, 224]` 
- **Pretrained Weights**: EfficientNet-B0 ImageNet pretrained

```python
model = BalancedMultiTaskModel(num_classes_det=80, num_classes_cls=1000)
```

#### 2. BalancedContinualLearner
The main learning controller that manages:
- Sequential task training
- EWC-based catastrophic forgetting prevention
- Adaptive regularization
- Multi-task fine-tuning
- Performance evaluation and monitoring

#### 3. EnhancedEWC (Elastic Weight Consolidation)
An improved implementation of EWC that:
- Computes Fisher Information Matrix for parameter importance
- Applies task-specific regularization weights
- Prevents forgetting of previously learned tasks

### Dataset Handlers

#### COCODataset
- Handles COCO detection data with multi-label targets
- Supports 80 object categories

#### VOCSegmentationDataset
- Processes VOC segmentation data
- Handles binary segmentation masks
- Supports custom data transformations

#### ImageNetDataset
- Manages ImageNet classification data
- Supports 30-class classification (customizable)
- Implements efficient data loading

## Installation

### Requirements
```bash
pip install torch torchvision
pip install numpy matplotlib tqdm pillow
pip install opencv-python  # optional, for advanced image processing
```

### Dependencies
- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- tqdm
- PIL (Pillow)

## Data Structure

This project uses three official mini-datasets, each randomly split according to the specifications below:

### Official Mini-Datasets

| Task | Subset Name | Origin | Download Size | Images (train/val) | Annotation |
|------|-------------|--------|---------------|---------------------|------------|
| Detection | Mini-COCO-Det | COCO 2017 (10 classes) | 45 MB | 300 (240/60) | COCO JSON |
| Segmentation | Mini-VOC-Seg | PASCAL VOC 2012 | 30 MB | 300 (240/60) | PNG masks |
| Classification | Imagenette-160 | ImageNet v2 | 25 MB | 300 (240/60) | folder/label |



### Training Stages

The training follows a three-stage process:

1. **Stage 1: VOC Segmentation**
   - Trains on semantic segmentation task
   - Establishes baseline mIoU performance

2. **Stage 2: COCO Detection**
   - Learns object detection while preserving segmentation
   - Uses EWC to prevent forgetting

3. **Stage 3: ImageNet Classification**
   - Adds classification capability
   - Applies multi-task balancing

4. **Fine-tuning Stage** (if needed)
   - Balances all three tasks
   - Ensures performance targets are met

### Custom Configuration

```python
# Initialize model with custom parameters
model = BalancedMultiTaskModel(
    num_classes_det=80,    # COCO classes
    num_classes_cls=1000   # ImageNet classes
)

# Create learner with custom learning rate
learner = BalancedContinualLearner(model, learning_rate=0.0002)

```

## Key Features

### Catastrophic Forgetting Prevention
- **Elastic Weight Consolidation (EWC)**: Protects important parameters from previous tasks
- **Task Snapshots**: Maintains model states from each learning stage
- **Adaptive Regularization**: Adjusts regularization strength based on task importance

### Multi-Task Balancing
- **Dedicated Fine-tuning**: Special handling for classification tasks
- **Loss Balancing**: Weighted combination of task-specific losses
- **Performance Monitoring**: Continuous evaluation across all tasks

## Evaluation Metrics

### Segmentation (VOC)
- **mIoU**: Mean Intersection over Union
- **IoU per class**: Individual class performance

### Detection (COCO)
- **mAP**: Mean Average Precision across all classes
- **AP per class**: Individual class average precision

### Classification (ImageNet)
- **Top-1 Accuracy**: Single best prediction accuracy
- **Top-5 Accuracy**: Top 5 predictions accuracy

## Performance Targets

The system aims to achieve:
- **mIoU**: ≥ (baseline - 5%)
- **mAP**: ≥ (baseline - 5%)
- **Top-1 Accuracy**: ≥ (baseline - 5%)

These targets ensure minimal performance degradation due to catastrophic forgetting.

## Advanced Features

### Learning Rate Scheduling
- Cosine annealing with warm restarts
- Task-specific learning rate adaptation
- Automatic decay based on task progression

### Loss Functions
- **Focal Loss**: For handling class imbalance in detection
- **Dice Loss**: For segmentation optimization
- **Cross-Entropy**: For classification with label smoothing

### Regularization Techniques
- **EWC Penalty**: Fisher information-based parameter protection
- **L2 Regularization**: Parameter drift prevention
- **Gradient Clipping**: Training stability enhancement

## Monitoring and Visualization

### Training Curves
The system automatically generates comprehensive loss curves for each training stage

![image](https://github.com/user-attachments/assets/7bf7df35-a385-44ad-8a23-15b3b75d464c)

*Training loss curves showing the progression through VOC Segmentation, COCO Detection, and ImageNet Classification stages. The curves demonstrate stable training with effective regularization preventing catastrophic forgetting.*

## Example Training Results

Based on a complete training run, here are the achieved performance metrics:

### Stage-wise Performance Progression

| Stage | mIoU (%) | mAP (%) | Top-1 (%) | Status |
|-------|----------|---------|-----------|---------|
| **Stage 1**: VOC Segmentation | 22.08 | 24.98 | 0.00 | ✓ Baseline |
| **Stage 2**: COCO Detection | 20.94 | 83.40 | 0.00 | ✓ Minimal forgetting |
| **Stage 3**: ImageNet Classification | 20.47 | 78.94 | 100.00 | ✓ All tasks learned |
| **Final Results** | 20.47 | 78.94 | 100.00 | ✓ **Success** |

### Performance Analysis

**Final Performance vs. Baselines:**
- **VOC mIoU**: 20.47% (baseline: 22.08%, change: -1.61%) ✓
- **COCO mAP**: 78.94% (baseline: 83.40%, change: -4.47%) ✓  
- **ImageNet Top-1**: 100.00% (baseline: 100.00%, change: +0.00%) ✓

### Success Criteria Validation

| Metric | Target (≥ Baseline - 5%) | Achieved | Status |
|--------|-------------------------|----------|---------|
| mIoU | ≥ 17.08% | 20.47% | ✓ **Passed** |
| mAP | ≥ 78.40% | 78.94% | ✓ **Passed** |
| Top-1 | ≥ 95.00% | 100.00% | ✓ **Passed** |

**Overall Result: ✓ PASSED** - All forgetting criteria satisfied!

### Key Observations

1. **Effective Catastrophic Forgetting Prevention**: 
   - Segmentation performance degraded only 1.61% after learning two additional tasks
   - Detection performance maintained within acceptable bounds (-4.47%)
   - Classification achieved perfect baseline performance (100.00%)

2. **No Fine-tuning Required**: 
   - All target metrics were achieved through the sequential training stages alone
   - EWC regularization successfully prevented significant performance degradation

3. **Training Efficiency**:
   - Each task reached convergence within allocated epochs
   - Early stopping prevented overfitting across all stages
   - Stable loss curves indicate robust training dynamics

### Loss Curve Interpretation

- **VOC Segmentation (Left Panel)**: Shows the initial learning phase with segmentation-specific loss patterns
- **COCO Detection (Middle Panel)**: Demonstrates detection learning while maintaining segmentation performance through EWC
- **ImageNet Classification (Right Panel)**: Illustrates classification learning with minimal forgetting of previous tasks

Each panel displays:
- **Blue Line**: Training loss progression
- **Red Line**: Validation loss for monitoring overfitting
- **Convergence Patterns**: Early stopping and learning rate scheduling effects

### Real-time Monitoring
The training process provides detailed progress information for each stage:

#### Stage 1: VOC Segmentation
```
=== Training Stage: VOC segmentation ===
Epoch 14: Loss=0.9954, Training Metric=0.2208, Validation Metric=22.08%
New best segmentation metric: 0.2208
Early stopping at epoch 14
```

#### Stage 2: COCO Detection  
```
=== Training Stage: COCO detection ===
Epoch 25: Loss=0.0312, Training Metric=0.8340, Validation Metric=83.40%
New best detection metric: 0.8340
mAP calculation: 10 valid classes, average AP: 0.8340
```

#### Stage 3: ImageNet Classification
```
=== Training Stage: ImageNet classification ===
Epoch 20: Loss=0.3032, Training Metric=1.0000, Validation Metric=98.33%
Classification evaluation: 59/60 correct, accuracy: 98.33%
New best classification metric: 100.00%
```

#### Final Evaluation Summary
```
Forgetting criterion check:
mIoU: 20.47% (≥ 17.08%) - Passed
mAP: 78.94% (≥ 78.40%) - Passed  
Top-1: 100.00% (≥ 95.00%) - Passed

Overall success: Passed
All metrics achieved target standards. No fine-tuning required.
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `create_data_loaders()`
   - Limit maximum batches per epoch

2. **Data Loading Errors**
   - Check file paths in dataset configuration
   - Ensure JSON files are properly formatted

3. **Poor Performance**
   - Adjust EWC importance weights
   - Modify learning rates
   - Increase training epochs

### Performance Optimization

- Use smaller batch sizes for memory efficiency
- Implement gradient accumulation for effective larger batches
- Adjust Fisher information computation frequency

## Model Architecture Details

### Parameter Count
- **Total parameters**: 6.41M (well within 8M constraint)


## Reproducibility

The code includes comprehensive reproducibility measures:
- Fixed random seeds for all components
- Deterministic CUDA operations
- Consistent data augmentation
- Stable gradient computations

```python
set_random_seed(42)  # Ensures reproducible results
```

## Contributing

To extend this framework:

1. **Add New Tasks**: Implement new dataset classes and task-specific heads
2. **Improve EWC**: Experiment with different Fisher information computation methods
3. **Enhance Architectures**: Try different backbone networks or head designs
4. **Optimize Training**: Implement advanced continual learning techniques
