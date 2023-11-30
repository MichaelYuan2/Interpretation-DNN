import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import logging
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

label_class = {'failed':0, 'alive':1}

# image data path
DATAPATH = r'american_bankruptcy.csv'

class CustomDataset():
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        return sample

def preprocess_data(df):
    index = []
    logging.info('Preprocessing data...')
    for i in tqdm(range(1, 8972)):
        company_name = 'C_{}'.format(i)
        company_data = df[df.company_name == company_name]
        max = company_data.year.max()
        min_year = max - 2
        df = df.drop(df[(df.company_name == company_name) & (df.year < min_year)].index)
    return df

def oversampling_data(df):
    X, y = df.drop(columns=['company_name', 'year', 'status_label']), df['status_label']

    # oversampling the minority with Nrm / Nm = 1
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_res['status_label'] = y_res
    return X_res

def load_data(TRAINPATH):
    df = pd.read_csv(TRAINPATH)
    df = preprocess_data(df)
    df = df.dropna()
    df['status_label'] = df['status_label'].map(label_class)
    df = oversampling_data(df)
    return df

def ratios_dataframe(df):
    # convert the dataframe to ratios
    ratios_df = pd.DataFrame()
    for column in df.columns:
        df[column] = df[column].replace(0, 1e-6)
    for i in range(18):
        for j in range(i+1, 18):
            column = "X{}/X{}".format(i+1, j+1)
            ratios_df[column] = df["X{}".format(i+1)] / df["X{}".format(j+1)]
            ratios_df[column] = (ratios_df[column] - ratios_df[column].mean()) / ratios_df[column].std() * 100 + 128
    ratios_df['status_label'] = df['status_label']
    return ratios_df

def create_dataset(df):
    dataset = []
    df = ratios_dataframe(df)
    logging.info('Creating dataset...')
    for index, data in tqdm(df.iterrows(), total=df.shape[0]):
        status_label = data['status_label']
        data = data.drop(columns=['status_label'])
        data = data.to_numpy()
        zeros_to_add = max(0, 169 - len(data))
        data = np.pad(data, (0, zeros_to_add), mode='constant', constant_values=0).reshape(13, 13)
        # data = rearrange_image(data)
        data = enlarge_image(data)
        data = torch.Tensor(data)
        data = data.unsqueeze(0)
        dataset.append((data, status_label))
    
    return CustomDataset(dataset)

def create_dataset_old(df):
    dataset = []
    logging.info('Creating dataset...')
    for index, data in tqdm(df.iterrows(), total=df.shape[0]):
        status_label = data['status_label']
        data = data.loc['X1':'X18']
        data = data.to_numpy()
        data = array_to_image(data)
        data = rearrange_image(data)
        data = enlarge_image(data)
        data = torch.Tensor(data)
        data = data.unsqueeze(0)
        dataset.append((data, status_label))
    
    return CustomDataset(dataset)

def array_to_image(array):
    image = np.zeros(169)
    p = 0
    for i in range(18):
        for j in range(i+1, 18):
            if array[j] != 0:
                image[p] = array[i]/array[j]
                p+=1
    image = image.reshape(13, 13)
    return image

def array_to_dataframe(array):
    df = pd.DataFrame(index=range(13), columns=range(13))
    for i in range(13):
        for j in range(13):
            df.iloc[i, j] = "X{}/X{}".format(i+1, j+1)
    return df

def rearrange_image(image):
    pixels = np.load(r"pixels.npy") # load the order of ratios from MC simulation
    image = image.reshape(-1)
    new_image = np.zeros(image.size)
    for i, pixel in enumerate(pixels):
        new_image[i] = image[pixel]
    return new_image.reshape(13, 13)

def enlarge_image(image_array, new_size=(64, 64)):
    """
    Enlarge an 13x13 image to 64x64 using nearest neighbor method.

    Args:
    image_array (numpy.ndarray): An 13x13 numpy array representing the image.
    new_size (tuple): New size for the image and dataframe, default is (64, 64).

    Returns:
    tuple: A tuple containing the enlarged image as a numpy array and the enlarged dataframe.
    """
    if image_array.shape != (13, 13):
        raise ValueError("Input image array must be 13x13 in size.")

    # Enlarge the image array
    image_pil = Image.fromarray(image_array)
    enlarged_image_pil = image_pil.resize(new_size, Image.NEAREST)
    enlarged_image_array = np.array(enlarged_image_pil)

    return enlarged_image_array

def enlarge_image_and_dataframe(image_array, dataframe, new_size=(64, 64)):
    """
    Enlarge an 13x13 image and a corresponding 13x13 dataframe to 64x64 using nearest neighbor method.

    Args:
    image_array (numpy.ndarray): An 13x13 numpy array representing the image.
    dataframe (pandas.DataFrame): An 13x13 pandas dataframe containing information for each pixel.
    new_size (tuple): New size for the image and dataframe, default is (64, 64).

    Returns:
    tuple: A tuple containing the enlarged image as a numpy array and the enlarged dataframe.
    """
    if image_array.shape != (13, 13) or dataframe.shape != (13, 13):
        raise ValueError("Input image array and dataframe must be 13x13 in size.")

    # Enlarge the image array
    image_pil = Image.fromarray(image_array)
    enlarged_image_pil = image_pil.resize(new_size, Image.NEAREST)
    enlarged_image_array = np.array(enlarged_image_pil)

    # Enlarge the dataframe
    enlarged_dataframe = pd.DataFrame(index=range(new_size[0]), columns=range(new_size[1]))
    scale_x, scale_y = new_size[0] / 13, new_size[1] / 13

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            original_i, original_j = int(i / scale_x), int(j / scale_y)
            enlarged_dataframe.iloc[i, j] = dataframe.iloc[original_i, original_j]

    return enlarged_image_array, enlarged_dataframe

def array_to_grayscale_image(array):
    """
    Converts a 64x64 numpy array into a grayscale image, labeling each row and column.
    Args:
    array (numpy.ndarray): A 64x64 numpy array.
    
    Returns:
    matplotlib.figure.Figure: A matplotlib figure representing the grayscale image.
    """
    if array.shape != (64, 64):
        raise ValueError("Input array must be 64x64 in size.")

    fig, ax = plt.subplots()
    # Displaying the image
    cax = ax.matshow(array, cmap='gray')
    # Adding color bar
    fig.colorbar(cax)

    # Adding row labels
    for i in range(64):
        ax.text(-1, i, str(i), va='center', ha='right', color='red')

    # Adding column labels
    for i in range(64):
        ax.text(i, -1, str(i), va='bottom', ha='center', color='blue')

    return fig

if __name__ == "__main__":
    # Example usage grayscale image:
    # Create a random 20x20 numpy array
    #test_array = np.random.rand(20, 20)
    # Convert to grayscale image
    #fig = array_to_grayscale_image(test_array)
    #plt.show()

    # Example usage dataframe and image:
    # Create a random 13x13 numpy array and a corresponding dataframe
    test_image_array = np.random.rand(13, 13) * 255
    test_dataframe = pd.DataFrame(np.random.choice(['Info1', 'Info2', 'Info3'], (13, 13)))

    # Enlarge both the image and the dataframe to 64x64
    enlarged_image, enlarged_dataframe = enlarge_image_and_dataframe(test_image_array, test_dataframe)


    # Display the first few rows of the enlarged dataframe for demonstration
    print(enlarged_dataframe.head(5))

    logging.basicConfig(level=logging.INFO)
    df = load_data(DATAPATH)
    dataset = create_dataset(df)
    print(dataset[0])