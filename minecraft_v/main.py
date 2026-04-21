import argparse

from minecraft_v.models import Netlist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True)
    args = parser.parse_args()

    with open(args.netlist) as f:
        netlist = Netlist.model_validate_json(f.read())

    print(netlist.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
