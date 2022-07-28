"""
Code to read in a json describing hyperparam sweep and create
scripts to execute each combination
"""

import os, sys
import logging
import argparse
import itertools
import json
from typing import *

logging.basicConfig(level=logging.INFO)


def params_to_cli_args(param_dict: Dict[str, Any]) -> str:
    """Format the params to CLI arguments"""
    tokens = []
    for k, v in param_dict.items():
        prefix = "-" if len(k) == 1 else "--"
        if isinstance(v, bool):  # For booleans, treat as flags
            if v:
                tokens.append(f"{prefix}{k}")
        else:
            tokens.append(f"{prefix}{k} {v}")
    # Manually add this since we can't provide this in the json
    tokens.append(f"--outdir {params_to_filename(param_dict)}")
    retval = " ".join(tokens)
    return retval


def params_to_filename(
    param_dict: Dict[str, Any],
    blacklist_tokens: Collection[str] = {"model", "blacklist", "config", "pretrained"},
) -> str:
    """
    Format the params (in key-value pairs (param, arg)) into a filename
    blacklist_tokens species the params that are excluded from filename (typically path-like)
    """
    tokens = []
    for k, v in param_dict.items():
        if k in blacklist_tokens:
            logging.warning(f"Excluding {k}: {v} from generated fname")
            continue
        assert " " not in k, f"Parameter cannot have space: {k}"
        if isinstance(v, str) and " " in v:  # Replace spaces in value
            v = v.replace(" ", "_")
        # This is (value, key) for historical readability reasons
        tokens.append(f"{v}_{k}")
    return "_".join(tokens)


def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("exec", type=str, help="Executable to run")
    parser.add_argument(
        "json_config", type=str, help="Config file specifying hyperparams"
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["shellargs", "json"],
        default="json",
        help="Write args as shell args or as json file",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=1, help="Number of replicates to run"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.getcwd(),
        help="Directory to write shell files",
    )
    parser.add_argument(
        "-t", "--template", type=str, default="", help="template to append to",
    )
    return parser


def main():
    """Run script"""
    args = build_parser().parse_args()
    assert os.path.isfile(args.exec)
    assert os.path.isfile(args.json_config)

    if not os.path.isdir(args.outdir):
        logging.info(f"Creating output directory: {args.outdir}")
        os.makedirs(args.outdir)

    # Read in the template
    header_lines = []
    if args.template:
        with open(args.template) as source:
            header_lines = [l.strip() for l in source]

    # Read in the hypeparameters to sweep over
    with open(args.json_config) as source:
        params = json.load(source)

    # Create the scripts
    param_combos = list(itertools.product(*params.values()))
    outdirs = []
    logging.info(f"Writing scripts for {len(param_combos)} parameter combinations")
    for p in param_combos:
        d = dict(zip(params.keys(), p))
        logging.debug(f"Writing script for {d}")
        # Create out direcotry name
        outdir_name = os.path.join(args.outdir, params_to_filename(d))
        assert outdir_name not in outdirs, f"Duplicated output dir: {outdir_name}"
        outdirs.append(outdir_name)

        if args.mode == "shellargs":
            # Build command
            cli_args = params_to_cli_args(d)
            cmd = f"python {os.path.abspath(args.exec)} {cli_args}"
            script_lines = header_lines + [cmd]

            # Write script
            script_name = outdir_name + ".sh"
            with open(script_name, "w") as sink:
                for line in script_lines:
                    sink.write(line + "\n")
        elif args.mode == "json":
            # Write a json of all the parameters
            d = dict(zip(params.keys(), p))
            in_json_fname = outdir_name + ".json"
            with open(in_json_fname, "w") as sink:
                json.dump(d, sink, indent=4)

            cmd = f"python {args.exec} {in_json_fname} --outdir {outdir_name}"
            script_lines = header_lines + [cmd]
            script_name = outdir_name + ".sh"
            with open(script_name, "w") as sink:
                for line in script_lines:
                    sink.write(line + "\n")

        else:
            raise ValueError(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
