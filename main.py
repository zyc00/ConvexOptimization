from task import tasks
from algorithms import algorithms

task = "img"
algorithm = "EALM"

assert task in tasks.keys()
assert algorithm in algorithms.keys()

tasks[task](algorithms[algorithm])
