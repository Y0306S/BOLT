from dataclasses import dataclass
from typing import Union


@dataclass
class QueryExecutionSpec:
    id: str
    query: str
    timeout_secs: float


@dataclass
class CompletedQuery:
    spec: QueryExecutionSpec
    elapsed_secs: float

    def __repr__(self):
        return f"CompletedQuery([{self.spec.id}] {self.spec.query} : {self.elapsed_secs})"


@dataclass
class TimedOutQuery:
    spec: QueryExecutionSpec
    elapsed_secs: float

    def __repr__(self):
        return f"TimedOutQuery([{self.spec.id}] {self.spec.query} : {self.elapsed_secs})"


@dataclass
class FailedQuery:
    spec: QueryExecutionSpec
    elapsed_secs: float
    error: Union[Exception, str]

    def __repr__(self):
        return f"FailedQuery([{self.spec.id}] {self.spec.query} : {self.error})"


QueryResult = Union[CompletedQuery, TimedOutQuery, FailedQuery]
