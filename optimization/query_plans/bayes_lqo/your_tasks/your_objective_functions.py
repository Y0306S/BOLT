"""Define your objecive function(s) here
Note: All code assumes we seek to maximize f(x)
If you want to instead MINIMIZE the objecitve, multiple scores by -1 in
your query_black_box() method
"""

import sys

from oracle.oracle import CompletedQuery, FailedQuery, TimedOutQuery, WorkloadInput, oracle
from workload.workloads import (
    OracleCodec,
    WorkloadSpec,
    # IMDB_WORKLOAD_SET,
    get_workload_set,
)

sys.path.append("../")


class ObjectiveFunction:
    """Objective function f, we seek x that MAXIMIZE f(x)"""

    def __init__(
        self,
    ):
        pass

    def __call__(self, x_list, timeouts_list=None):
        """Input
            x_list:
                a LIST of input space items from the origianl input
                search space (i.e. list of aa seqs)
            timeouts_list:
                option list of timeout values for each item in x_list
                amount of time we run oracle on that x before quitting and
                keeping it as a censored observation
        Outputs
            scores_list:
                a LIST of float values obtained by evaluating your
                objective function f on each x in x_list
                or np.nan in the wherever x is an invalid input
            censoring_list:
                a LIST of int (0/1) values indicating whether
                or not each observation is censored
                (or a list of all 0's if the we are not censoring observations)
        """
        if timeouts_list is None:
            scores_list, censoring_list = self.query_black_box(x_list)
        else:
            scores_list, censoring_list = self.query_black_box(x_list, timeouts_list)
        return scores_list, censoring_list

    def query_black_box(self, x_list):
        """Input
            x_list:
                a LIST of input space items from the origianl input
                search space (i.e. list of aa seqs)
        Outputs
            scores_list:
                a LIST of float values obtained by evaluating your
                objective function f on each x in x_list
                or np.nan in the wherever x is an invalid input
            censoring_list:
                a LIST of int (0/1) values indicating whether
                or not each observation is censored
                (or a list of all 0's if the we are not censoring observations)
        """
        raise NotImplementedError("Must implement method query_black_box() for the black box objective")


VERBOSE_CALLBACK = False


def progress_callback(task_result):
    # Query completed successfully
    if isinstance(task_result, CompletedQuery):
        if VERBOSE_CALLBACK:
            print(f"SUCCESS: {task_result.spec.id} | {task_result.elapsed_secs} secs")
    # Query timed out
    elif isinstance(task_result, TimedOutQuery):
        if VERBOSE_CALLBACK:
            print(f"TIMEOUT: {task_result.spec.id}")
    # Query failed
    elif isinstance(task_result, FailedQuery):
        if VERBOSE_CALLBACK:
            print(f"FAILED: {task_result.spec.id}")
    # Unknown result type pls dont actually happen ;-;
    else:
        raise ValueError(f"Unknown task result type: {type(task_result)}")

    # Dictionary mapping from query id (string) to new timeout in seconds (float)
    return {}


class DatabaseObjective(ObjectiveFunction):
    """Database Oracles"""

    def __init__(
        self,
        workload_name,
        worst_runtime_observed=200,  # worst runtime from init data, used to set score for OOM query plans
        timeout=100,
        which_language="aliases",
        oracle_max_bsz=10,
        so_future=None,
    ):
        super().__init__()
        # provision instances for new run
        self.oracle_max_bsz = oracle_max_bsz
        self.which_language = which_language
        self.worst_runtime_observed = worst_runtime_observed
        self.timeout = timeout
        self.workload_name = workload_name
        workload_set_id = workload_name.split("_")[0]
        if workload_set_id == "CEB":
            workload_set_id = "CEB_3K"  # FOR NOW ASSUME CEB 3K, not 13K
        elif workload_set_id == "STACK":
            assert so_future is not None, "Must provide so_future for STACK workload"
            if so_future:
                workload_set_id = "SO_FUTURE"
            else:
                workload_set_id = "SO_PAST"
        WORKLOAD_SET = get_workload_set(workload_set=workload_set_id)  # JOB, CEB_3K, CEB_13K
        self.workload_spec = WORKLOAD_SET.queries[workload_name]
        self.full_workload_spec = WorkloadSpec.from_definition(
            definition=self.workload_spec,
            codec=self.get_codec(),
        )

    def get_codec(
        self,
    ):
        """Sets var self.codec to specific codec for query language"""
        if self.which_language == "join-order":
            codec = OracleCodec.JoinOrder
        elif self.which_language == "join-order-operators":
            codec = OracleCodec.JoinOrderOperators
        elif self.which_language == "aliases":
            codec = OracleCodec.Aliases
        else:
            assert 0, f"unknown lanugage: {self.which_language}"
        return codec

    def set_codec(
        self,
    ):
        """Sets var self.codec to specific codec for query language associated with oracle"""
        assert NotImplementedError, "Must implement set_codec to set self.codec for specific DB Oracle"

    def query_black_box(self, x_list, timeouts_list=None):
        if timeouts_list is None:
            timeouts_list = [self.timeout] * len(x_list)

        self.total_non_parallel_runtime = 0.0

        workloads = []
        for i in range(len(x_list)):
            encoded_query = x_list[i]
            timeout_secs = timeouts_list[i]

            wl_input = WorkloadInput(
                id=f"{i + 1}",
                encoded_query=encoded_query,
                timeout_secs=timeout_secs,
            )
            workloads.append(wl_input)

        results = oracle(
            workload=self.full_workload_spec,
            workload_inputs=workloads,
        )

        scores_list, censoring_list = [], []
        for result in results:
            runtime = result.elapsed_secs  # should be ~same as self.timeout for timed-out queries
            score = -abs(runtime)

            self.total_non_parallel_runtime += runtime

            if isinstance(result, CompletedQuery):
                censoring_list.append(0)
            elif isinstance(result, TimedOutQuery):
                censoring_list.append(1)
            elif isinstance(result, FailedQuery):
                censoring_list.append(1)
                # Failures should get self.timeout*2 N seconds and be censored data
                #   this way they are counted as oracle calls still and oracle can learn that they are bad data points
                # score = timeout_secs * -2
                ## Note: above creates an issue for non-timeout where we actually use very large timeout value...
                ## So instead let's make this no worse than the worst runtime observed in our init data
                bad_runtime = min(self.worst_runtime_observed, result.spec.timeout_secs * 2)
                score = bad_runtime * -1
            else:
                assert 0, "result is not an instance of one of expected types"

            scores_list.append(score)

        return scores_list, censoring_list


"""Objective functions with unique string identifiers 
identifiers can be passed in when running LOL-BO --task_id arg
whcih specifies which objective function to use 
--task_specific_args can be used to specify a list of args 
passed into the init of any of these objectives when they are initialized 
"""
OBJECTIVE_FUNCTIONS_DICT = {
    "db": DatabaseObjective,
}


if __name__ == "__main__":
    # Test new oracle_for_workload_cluster
    fake_vae_strings = [[0, 0, 0], [0, 1, 2], [0], [0, 0, 0, 0, 0, 0], [0, 3]]
    obj = OBJECTIVE_FUNCTIONS_DICT["db"](workload_name="CEB_1A10", worst_runtime_observed=200)
    scores_list, censoring_list = obj.query_black_box(x_list=fake_vae_strings, timeouts_list=[100, 100, 100, 100, 100])
    print(scores_list, censoring_list)
