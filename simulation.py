"""Run the assignment simulation and print mean/max errors."""

from pprint import pprint
from lns import run_suite, LNS8, LNS16


def main():
    print("LNS format ranges")
    for fmt in (LNS8, LNS16):
        print(
            f"{fmt.name}: log2 range [{fmt.min_log}, {fmt.max_log}], "
            f"magnitude range [{fmt.min_positive:.6g}, {fmt.max_magnitude:.6g}]"
        )

    print("\nSimulation results")
    pprint(run_suite(seed=7, n_random=10000))


if __name__ == "__main__":
    main()
