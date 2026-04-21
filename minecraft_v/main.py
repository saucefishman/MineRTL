import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True)
    args = parser.parse_args()

    with open(args.netlist) as f:
        netlist = json.load(f)

    print(json.dumps(netlist, indent=2))


if __name__ == "__main__":
    main()
