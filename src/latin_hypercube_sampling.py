import random

"""
given N hyperparameters
we divvy the range of each param into M segments
for each parameter we randomly sample without replacement from the segments
once we've done this M times we've segmented the space and evaluate everything


dict of param name and range
"""


def latin_hypercube_sampling(hpr, partitions):
    points = [dict() for _ in range(partitions)]
    for hp, (mn, mx, tp) in hpr.items():
        incr = (mx - mn) / partitions
        indices = random.sample(range(partitions), partitions)
        for i in range(0, partitions):
            points[indices[i]][hp] = tp(mn + incr * i + random.uniform(0, incr))

    return points


if __name__ == "__main__":
    d = {1: (1, 2), 2: (3, 4)}
    print(latin_hypercube_sampling(d, 2))
