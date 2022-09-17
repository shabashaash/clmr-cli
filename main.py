import argparse
from pytorch_lightning import Trainer
from classifier_script import main as cmain
from recomender_script import main as rmain

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="foobar-cli")

    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if args.model == "classes":
        cmain(args)
    elif args.model in ['full', 'recomend']:
        rmain(args)
    else:
        raise NotImplementedError("Model not implemented")