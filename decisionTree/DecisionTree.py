import numpy
import copy


class DecisionTree(object):

    def __init__(self, attribute_names):
        self.tree = {}
        self.attribute_names = list(attribute_names)
        self.attribute_order = list(attribute_names)

    def set_attribute_names(self, data):
        attribute_names = {}
        for name in self.attribute_names:
            attribute_names[name] = []
        for instance in data:
            for i in range(len(instance) - 1):
                if instance[i] not in attribute_names[self.attribute_names[i]]:
                    attribute_names[self.attribute_names[i]].append(instance[i])
        self.attribute_names = attribute_names

    def frequency_counter(self, data, col):
        # Check to see if invalid column index or if data is empty
        if len(data) == 0 or len(data[0]) - 1 < col:
            return "ERROR"

        # Create a frequency dictionary for items in the columns of data
        dict = {}
        for val in data:
            if val[col] in dict:
                dict[val[col]] += 1
            else:
                dict[val[col]] = 1

        # Determine the most frequent item in the specified column
        most_frequent = 0
        most_common = None
        for key, value in dict.items():
            if value > most_frequent:
                most_common = key
                most_frequent = value

        # Return the most common value in the specified column
        return most_common

    def calc_entropy(self, data):
        num_instances = len(data)

        # Calculate the entropy for the given attribute
        entropy_dict = {}
        for instance in data:
            if instance[0] in entropy_dict:
                if instance[1] in entropy_dict[instance[0]]:
                    entropy_dict[instance[0]][instance[1]] += 1
                else:
                    entropy_dict[instance[0]][instance[1]] = 1
            else:
                entropy_dict[instance[0]] = {}
                entropy_dict[instance[0]][instance[1]] = 1

        entropy = 0
        for key, value in entropy_dict.items():
            instances_per_branch = 0
            for key2, value2 in value.items():
                instances_per_branch += value2

            # Find the log value and the weight of each branch of the current attribute
            log_val = 0
            for key2, value2 in value.items():
                # TODO: Make sure it is doing floating point arithmetic
                p = value2 / instances_per_branch
                log_val += -p * numpy.log2(p)
            weight = instances_per_branch / num_instances
            entropy += log_val * weight

        return entropy

    def print_tree(self, tree, recursions):
        recursions += 1
        if isinstance(tree, list) and isinstance(tree[0], list):
            for i in range(len(tree)):
                self.print_tree(tree[i], recursions)
            return

        if not isinstance(tree, dict):
            print("\t" * recursions, tree)
            return

        for key, value in tree.items():
            print("\t" * recursions, key)
            self.print_tree(value, recursions)

    # Assume data is a 2 dimensional array, default is the most common target value in the total training set
    def make_tree(self, data, attribute_names, default):
        # Store the number of instances and attributes left in the data
        nInstances = len(data)

        # Check to see if all the targets in the dataset are the same
        same_targets = True
        if len(data) > 0:
            # Check to see if all of the targets are the same
            targ = data[0][len(data[0]) - 1]
            for value in data:
                if targ != value[len(value) - 1]:
                    same_targets = False
                    break

        if nInstances == 0 or len(data[0]) == 1:
            return default
        elif same_targets:
            return data[0][len(data[0]) - 1]
        else:
            # Iterate through each attribute and calculate the entropy
            entropies = []
            for i in range(len(data[0]) - 1):
                attr_data = []
                for attribute in data:
                    inner_arr = []
                    inner_arr.append(attribute[i])
                    inner_arr.append(attribute[len(data[0]) - 1])
                    attr_data.append(inner_arr)
                entropies.append(self.calc_entropy(attr_data))

            # Create new parameters for next round
            best_attribute_index = entropies.index(min(entropies))
            tree = {}
            tree["name"] = attribute_names[best_attribute_index]
            new_data_array = copy.deepcopy(data)
            new_attribute_names = list(attribute_names)
            del new_attribute_names[best_attribute_index]
            for instance in new_data_array:
                del instance[best_attribute_index]
            for i in range(len(data)):
                if data[i][best_attribute_index] not in tree:
                    tree[data[i][best_attribute_index]] = [new_data_array[i]]
                else:
                    tree[data[i][best_attribute_index]].append(new_data_array[i])

            # Recursively call the make_tree function
            for key, value in tree.items():
                if key != "name":
                    tree[key] = self.make_tree(value, new_attribute_names, default)

            # Check to see if the subtree has all of the possible decisions in its children
            for possible_value in self.attribute_names[tree["name"]]:
                if possible_value not in tree:
                    tree[possible_value] = default

            # If all values in the subtree are all the default value, then simply make that subtree a leaf node
            all_same = True
            for key, value in tree.items():
                if key != "name" and value != default:
                    all_same = False
            if all_same:
                tree = default

            return tree

    def train(self, data):
        if len(data) > 0:
            default = self.frequency_counter(data, len(data[0]) - 1)
            attribute_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            self.set_attribute_names(data)
            self.tree = self.make_tree(data, attribute_names, default)
            self.print_tree(self.tree, -1)

    def predict(self, data):
        predictions = []
        for instance in data:
            traversal_tree = copy.deepcopy(self.tree)
            while isinstance(traversal_tree, dict):
                index = self.attribute_order.index(traversal_tree["name"])
                traversal_tree = traversal_tree[instance[index]]
            predictions.append(traversal_tree)

        return predictions
