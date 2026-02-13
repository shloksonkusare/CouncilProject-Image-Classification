# CouncilProject – Image Classification
## Dataset Sources

Please download the datasets from the following links:

🔗 Road & Garbage Dataset:
https://data.mendeley.com/datasets/zndzygc3p3/2

🔗 Street Light Dataset (Electricity Issues):
https://github.com/Team16Project/Street-Light-Dataset

## Required Folder Structure

After downloading, organize the images in the following directory structure:

```
ImageData/
│
├── DamagedRoads/
├── ElectricityIssues/
└── GarbageAndSanitation/
```

Each folder must contain images belonging only to that specific class.

## Dataset Preparation Guidelines

To ensure good model performance:

Keep a balanced dataset
→ Maintain approximately equal number of images in each class.

Remove:
- Duplicate images

- Corrupted files

- Irrelevant images

Ensure:

- Images are clear and properly labeled

- Classes do not overlap

## Important Notes

Folder names must match exactly:
```
DamagedRoads

ElectricityIssues

GarbageAndSanitation
```
The training script automatically reads folder names as class labels.

Incorrect folder names will result in classification errors