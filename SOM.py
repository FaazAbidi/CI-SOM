import numpy as np
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from hexalattice.hexalattice import *
import itertools
import random

# generic implementation of Self Organizing Maps
class SOM:
    def __init__(self, map_size: tuple, learning_rate: int, epochs: int)-> None:
        """
        Initialize the SOM with the given parameters and random weights.
        """
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        map_shape = map_size[0], map_size[1], 3
        self.map = np.random.random(map_shape)

        self.map_radius = max(self.map_size[0], self.map_size[1])/2
        self.time_constant = self.epochs/np.log(self.map_radius)

        self._2d_coordinates_map = self._get_2d_coordinates_grid(self.map_size)

    def train(self, training_data):
        """
        Start the training of the SOM with the given input vector and plotiate the map after each epoch.
        """
        print("Started training...")
        for epoch in tqdm(range(self.epochs)):
            # get learning rate
            learning_rate = self._learning_rate_function(time=epoch)
            # get neighborhood radius
            neighborhood_radius = self._neighborhood_radius(time=epoch)
            # iterating over all training data and train the SOM
            for data in training_data:
                self._train_on_element(data, learning_rate, neighborhood_radius)
            self.plot_map(epoch)
            
    
    def _train_on_element(self, element, learning_rate, neighborhood_radius):
        
        distance_map = np.linalg.norm(element - self.map, axis=2)
        # Get index of weight closest to training element
        winner = np.unravel_index(np.argmin(distance_map), distance_map.shape)
        # Get neighborhood function
        neighborhood_function = self.neighbourhood_function(winner, neighborhood_radius)
        # Stack neighborhood function 3 times in z-direction, so weights map can be multiplied by it
        neighborhood_function_replicated = np.dstack(
            (neighborhood_function, neighborhood_function, neighborhood_function))

        weight_change = neighborhood_function_replicated * learning_rate * (element - self.map)
        self.map += weight_change

    def BMU(self, vector):
        """
        Compute the Best Matching Unit of the SOM with the given input vector.
        """
        # calculate the distances between the input vector and the weights
        distances = np.linalg.norm(self.weights - vector, axis=1)
        # choose the index of the minimum distance
        winner = np.argmin(np.sum(distances, axis=1))
        return winner
    
    def neighbourhood_function(self, winner, radius):
        """
        Guassian function for the dynamic neighbourhood function. This functions returns a 2D array of the size of the map
        and values in the array are of learning rate multiplied by the guassian function or neighborhood function.
        """
        distance_to_winner = np.linalg.norm(winner - self._2d_coordinates_map, axis=2)
        exponent = -1 * (distance_to_winner / (2 * radius**2))**2
        return np.exp(exponent)

    def _neighborhood_radius(self, time):
        """
        Radius of the neighborhood function for the dynamic neighborhood function. This radius will decay with time.
        """
        return self.map_radius * np.exp(-time/self.time_constant)

    def _learning_rate_function(self, time):
        """
        Learning rate function for the dynamic learning rate. This learning rate will decay with time.
        """
        return self.learning_rate * np.exp(-time/self.epochs)


    def _get_2d_coordinates_grid(self, shape):
        """
        Given a 2 element tuple, return a 2D matrix of grid coordinates.
        Each element of the grid is a point [x, y] where x and y are coordinate values
        """

        y = range(0, shape[0])
        x = range(0, shape[1])

        yx_list = list(itertools.product(y, x))
        yx_vector = np.array(yx_list)

        yx_matrix = np.array(yx_vector).reshape((shape[0], shape[1], 2))
        return yx_matrix


    def plot_map(self, epoch):
        """
        Plots the map of the SOM.
        """
        hex_centers, _ = create_hex_grid(nx=self.map_size[0],
                                 ny=self.map_size[1],
                                 do_plot=False)

        x_hex_coords = hex_centers[:, 0]
        y_hex_coords = hex_centers[:, 1]

        colors = self.map.reshape(self.map_size[0]*self.map_size[1],3)

        plot_single_lattice_custom_colors(x_hex_coords, y_hex_coords,
                                      face_color=colors,
                                      edge_color=colors,
                                      min_diam=1.,
                                    plotting_gap=0,
                                    rotate_deg=0,
                                    line_width=0.5,)
        # plt.savefig('SOM_iterations/som_map_{}.png'.format(epoch))
        # plt.show()