"""Define your black box constraints function(s) here
should take in a input space (x) value a constraint value
such that the constraint is of the from c(x) <= 0
"""

import sys

sys.path.append("../")
import torch


class ConstraintFunction:
    """Constraint function in form of c(x) <= 0"""

    def __init__(
        self,
        threshold_value,
        threshold_type,  # is the threshold a min allowed or max allowed value ?
    ):
        assert threshold_type in ["min", "max"]
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value

    def __call__(self, x_list):
        """Input
            x_list: a list of input space items x_list
        Output
            c_vals
                torch tensor of shape (len(x_list), 1)
                Returns torch tensor of associated constraint values c_vals
                which are values converted from raw function values to values
                such that the constraint is of the form c(x) <= 0

        """
        c_vals = self.query_black_box(x_list)
        if self.threshold_type == "min":
            # convert to format such that c(x) <= 0
            # for min threshold we do threshold - cval
            c_vals = self.threshold_value - c_vals
        elif self.threshold_type == "max":
            # convert to format such that c(x) <= 0
            # for max threshold we do cval - threshold
            c_vals = c_vals - self.threshold_value

        return c_vals.unsqueeze(-1)

    def query_black_box(self, x_list):
        """Input
            x_list: a list of input space items x_list
        Output
            c_func_values:
                a tensor of shape (len(x_list),)
                has associated raw constraint function values for each x in x_list
        """
        raise NotImplementedError("Must implement method query_black_box() for the black box constraint")


class ExampleLengthConstraint(ConstraintFunction):
    """Example constraint constraining length of the input space items"""

    def __init__(
        self,
        threshold_value,
        threshold_type,  # is the threshold a min allowed or max allowed value ?
    ):
        super().__init__(
            threshold_type=threshold_type,
            threshold_value=threshold_value,
        )

    def query_black_box(self, x_list):
        c_vals = []
        for x in x_list:
            c_vals.append(len(x))

        return torch.tensor(c_vals).float()


class ExampleNumGsConstraint(ConstraintFunction):
    """Example constraint constraining number of G's in input seqs
    (assuming here that xs are strings)
    """

    def __init__(
        self,
        threshold_value,
        threshold_type,  # is the threshold a min allowed or max allowed value ?
    ):
        super().__init__(
            threshold_type=threshold_type,
            threshold_value=threshold_value,
        )

    def query_black_box(self, x_list):
        # compute number of gs in each input sequence and return as tensor
        if not type(x_list) == list:
            x_list = x_list.tolist()
        c_vals = []
        for x in x_list:
            if len(x) == 0:
                c_vals.append(0.0)
            else:
                num_gs = 0
                for char in x:
                    if char == "G":
                        num_gs += 1
                c_vals.append(num_gs)

        return torch.tensor(c_vals).float()


CONSTRAINT_FUNCTIONS_DICT = {}
CONSTRAINT_FUNCTIONS_DICT["length"] = ExampleLengthConstraint
CONSTRAINT_FUNCTIONS_DICT["num_gs"] = ExampleNumGsConstraint


if __name__ == "__main__":
    # Example constraint function for min length of 2
    cfunc = ExampleLengthConstraint(threshold_value=2, threshold_type="min")
    example_list = ["ABC", "DEF", "AA", "A"]
    values = cfunc(example_list)
    print(values)
