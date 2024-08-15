# Progressive-Transfer-Learning-for-Railway-Track-Fault-Detection
A semi-supervised model, which leverages tranfer learning and progressive training to effectively utilize the labeled dataset, to detect railway track faults. Initially trained on just 60 labeled data points, the model progressively selects and incorporates highly-confident data with pseudo-labels from the unlabeled dataset. 

## Project Overview

Railway track faults pose significant risks to transportation safety. Detecting these faults efficiently and accurately, and that in less amount of labeled datapoints, is critical. In this project, I use a combination of image augmentation and progressive transfer learning to build a model capable of identifying faults on railway tracks with high confidence, under way less labeled data than a fully supervised model.

## Methodology

### 1. Data Collection
- **Dataset**: The initial dataset is taken from Kaggle. [Click here](https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection) to get the dataset.
  
### 2. Image Augmentation
- **Data Generation**: I start by generating and saving around 600 unlabeled images of railway tracks from the available dataset using various image augmentation techniques. These images are stored in the `augmented_image` directory. Check the [imageAugmentation](https://github.com/baranwalayush/Progressive-Transfer-Learning-for-Railway-Track-Fault-Detection/blob/main/imageAugmentation.ipynb) notebook for more details.
- **Purpose**: The augmented_images will be used in progressive training of the model. Also augmentation helps in diversifying the dataset, simulating different scenarios that the model might encounter in real-world situations.

### 3. Transfer Learning & Initial Model Training
- **Model Architecture**: The model is based on a fine-tuned ResNet50, followed by a fully connected layer with 64 units, and a softmax output layer with 2 units (indicating 'Defective' and 'Not defective'). BatchNormalization and Dropout layers are also used in between the layers to tackle overfitting, which the model experienced during the previous runs.
- **Training on Labeled Data**: Initially, the model is trained on 60 labeled data points to learn the basic feature representations.

### 4. Progressive Training with Pseudo-Labeling
- **Pseudo-Labeling**: After the initial training, the model begins to make predictions on the unlabeled augmented images. Images with high-confidence predictions (output probability > 0.85) are pseudo-labeled and incorporated into the training dataset.
- **Progressive Incorporation**: The model is retrained progressively as more high-confidence pseudo-labeled images are added. This process continues until all augmented images are exhausted. Check [main_file](https://github.com/baranwalayush/Progressive-Transfer-Learning-for-Railway-Track-Fault-Detection/blob/main/main_file.ipynb) notebook for more details.
- **Objective**: The goal is to enhance the model's performance by utilizing a large amount of unlabeled data effectively, refining the model iteratively as it learns from more data. In doing so, the model firstly adopts easy and reliable pseudo-labeled spoofing data for model update and then explores difficult ones as the its discriminativeness ability improves.

## Requirements
- Python 3.x
- TensorFlow or PyTorch (Depending on Implemetation)
- CV2 for image processing
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
  
## How to Run
- **Get the Data**: Get the data from kaggle, using the link provided
- **Data Augmentation**: Run the data augmentation script to generate the unlabeled dataset.
- **Initial Training**: Train the model on the labeled dataset using the provided script.
- **Progressive Training**: Execute the progressive training script to iteratively include pseudo-labeled data and retrain the model.
- **Evaluation**: Once training is complete, evaluate the model's performance on a separate test dataset.

## References
This project is based on the research paper: [Progressive Transfer Learning for Face Anti-Spoofing](https://yu-wu.net/pdf/TIP2021_antispoof.pdf) by Ruijie Quan, Yu Wu, Xin Yu, and Yi Yang, Senior Member, IEEE. The methodology has been adapted for the detection of railway track faults.

## Acknowledgments
Thanks to the authors of the original paper for their innovative approach to progressive learning, which served as the foundation for this project.
Additional thanks to the open-source community for providing the tools and libraries used in this project.
