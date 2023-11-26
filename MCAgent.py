import pandas as pd
import numpy as np
from itertools import combinations
import logging
from math import sqrt, ceil
import tqdm

import random
import seaborn as sns
import matplotlib.pyplot as plt

from create_dataset import load_data, array_to_image, enlarge_image_and_dataframe, array_to_grayscale_image


class MCAgent():
    """
    Use pass the calculating best template for pixel
    """
    def __init__(self, data, labels) -> None:
        self.data = data # data is a dataframe
        self.ratios = data.columns
        self.labels = labels.reshape(-1)

        self.n_ratios = len(self.ratios)
        self.max_iterations = 3 * self.n_ratios
        self.print_every = 100
        self.pixels = []
        

    # Function to perform Monte Carlo simulation
    def monte_carlo_simulation(self):
        print("Start monte_carlo_simulation")
        print("N ratios: ", self.n_ratios)
        pixels = list(range(self.n_ratios))
        self.calculate_correlation()
        
        # Initial random assignment of financial ratios to pixels
        random.shuffle(pixels)
        current_energy = self.calculate_energy(pixels)
        
        iter_count = 0
        no_improvement_count = 0
        
        while no_improvement_count < self.max_iterations:
            # Randomly select two pixels to swap
            pixel1, pixel2 = random.sample(pixels, 2)
            
            # Swap the financial ratios corresponding to the selected pixels
            pixels[pixel1], pixels[pixel2] = pixels[pixel2], pixels[pixel1]
            
            # Calculate the new energy after the swap
            new_energy = self.calculate_energy(pixels)
            
            # If energy is reduced, accept the swap
            if new_energy < current_energy:
                current_energy = new_energy
                no_improvement_count = 0
            else:
                # Revert the swap
                pixels[pixel1], pixels[pixel2] = pixels[pixel2], pixels[pixel1]
                no_improvement_count += 1

            iter_count += 1
            if iter_count % self.print_every == 0:
                print("Iteration: {}, Energy: {}".format(iter_count, new_energy))
            
        new_labels = self.arrange_by_pixels(pixels)
        
        return new_labels, pixels

    # Arrage ratios and labels according to the pixels
    def arrange_by_pixels(self, pixels):
        # new_ratios = np.zeros(self.ratios.size)
        new_labels = np.full(self.labels.size, " ", dtype="S6")
        for i, pixel in enumerate(pixels):
            # new_ratios[i] = self.ratios[pixel]
            new_labels[i] = self.labels[pixel]

        # new_ratios = new_ratios.reshape(self.ratios.shape)
        new_labels = new_labels.reshape(13, 13)
        return new_labels
        

    # Function to calculate correlation coefficients for given financial ratios
    def calculate_correlation(self):
        correlation_matrix = self.data.corr()

        # Create a dictionary to store correlations between each pair of ratios
        self.correlation_dict = {}

        # Loop through the pairs of ratios in the correlation matrix
        for ratio1 in correlation_matrix.columns:
            for ratio2 in correlation_matrix.columns:
                # Skip comparing a ratio with itself
                if ratio1 != ratio2:
                    correlation_value = correlation_matrix.loc[ratio1, ratio2]
                    ratio_pair = (ratio1, ratio2)
                    self.correlation_dict[ratio_pair] = correlation_value
        # print(self.correlation_dict)

    def get_correlation(self, ratio1, ratio2):
        return self.correlation_dict[(ratio1, ratio2)]

    # Function to calculate the energy (objective function) for the given assignment of financial ratios to pixels
    def calculate_energy(self, pixels):
        energy = 0
        
        for pixel1, pixel2 in combinations(pixels, 2):
            ratio1 = self.ratios[pixel1]
            ratio2 = self.ratios[pixel2]
            
            correlation = self.get_correlation(ratio1, ratio2)
            if correlation is not None:
                new_energy = abs(correlation) * self.calculate_distance(pixel1, pixel2)
                energy += new_energy if not np.isnan(new_energy) else 0
                
        return energy


    # Function to calculate Euclidean distance between two pixels
    def calculate_distance(self, pixel1, pixel2):
        x1, y1 = divmod(pixel1, self.n_ratios)
        x2, y2 = divmod(pixel2, self.n_ratios)
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    