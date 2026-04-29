import argparse

import monai.data.meta_obj

from diffae.experiment import train
from diffae.templates import templates_dict

monai.data.meta_obj.set_track_meta(False)

if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "template",
        type=str,
        help="template to use for training",
    )

    parser.add_argument(
        "--conf",
        type=str,
        required=False,
        help="name of the experiment, used to load additional information from json ",
        default=None,
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run fast_dev_run without initializing wandb",
    )

    # parse arguments
    args = parser.parse_args()
    # create config
    conf = templates_dict[args.template](args.conf)
    train(conf, fast_dev_run=args.fast_dev_run)
