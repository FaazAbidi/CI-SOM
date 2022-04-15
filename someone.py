import numpy as np
import itertools
import cv2


def get_2d_coordinates_grid(shape):
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


class SOCMTrainer:
    """
    A class for training a Self Organizing Color Map
    """

    def __init__(self, shape, iterations_number):
        """
        :param shape: a 2 elements list that specifies map size
        :param iterations_number: number of iterations map will be trained on input
        """
        if len(shape) != 2:
            raise ValueError("Shape must have 2 dimensions")

        # Each color has 3 elements
        map_shape = shape + [3]

        self.map = np.random.random(map_shape)
        print(self.map.shape)
        self.iterations_number = iterations_number

        self.coordinates_grid = get_2d_coordinates_grid(self.map.shape)

    def train(self, training_data):

        for iteration_index in range(self.iterations_number):

            print("Iteration {}/{}".format(iteration_index, self.iterations_number))

            # Learning rate and neighborhood width don't depend on training data
            learning_rate = self._get_learning_rate(iteration_index)
            neighborhood_width = self._get_neighborhood_function_width(iteration_index)

            for element in training_data:
                self._train_on_element(element, learning_rate, neighborhood_width)

            # Show map on current iteration, scaled up quite a bit to be visible
            cv2.imshow("map", cv2.pyrUp(self.map))
            cv2.waitKey(30)

    def _train_on_element(self, element, learning_rate, neighborhood_width):

        # Compute distances from element to map weights
        distance_map = np.linalg.norm(element - self.map, axis=2)

        # Get index of weight closest to training element
        best_matching_unit_indices = np.unravel_index(np.argmin(distance_map), distance_map.shape)

        neighborhood_function = self._get_neighborhood_function(best_matching_unit_indices, neighborhood_width)

        # Stack neighborhood function 3 times in z-direction, so weights map can be multiplied by it
        neighborhood_function_replicated = np.dstack(
            (neighborhood_function, neighborhood_function, neighborhood_function))

        weight_increment = neighborhood_function_replicated * learning_rate * (element - self.map)
        self.map += weight_increment

    def _get_neighborhood_function(self, best_matching_unit_indices, neighborhood_width):

        # Get distances from best match coordinate to other coordinates
        # print(self.coordinates_grid.shape)
        # print(self.coordinates_grid)
        distance_to_best_match = np.linalg.norm(best_matching_unit_indices - self.coordinates_grid, axis=2)

        exponent = -1 * (distance_to_best_match / (2 * neighborhood_width**2))**2
        neighborhood_function = np.exp(exponent)

        return neighborhood_function

    def _get_learning_rate(self, iteration_number):

        base_width = max(self.map.shape[0], self.map.shape[1]) / 2
        time_base = self.iterations_number / np.log(base_width)

        learning_rate = 0.1 * np.exp(-1 * iteration_number / time_base)
        return learning_rate

    def _get_neighborhood_function_width(self, iteration_number):

        base_width = max(self.map.shape[0], self.map.shape[1]) / 2
        time_base = self.iterations_number / np.log(base_width)

        neighborhood_width = base_width * np.exp(-1 * iteration_number / time_base)
        return neighborhood_width


if __name__ == "__main__":

    socm_trainer = SOCMTrainer([200, 200], 100)
    training_samples = np.random.random([100, 3])
    socm_trainer.train(training_samples)

    map = cv2.pyrUp(socm_trainer.map)

    cv2.imshow("map", map)
    cv2.waitKey(0)