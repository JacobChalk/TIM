from time_interval_machine.utils.parser import parse_args
from time_interval_machine.utils.misc import launch_job
from scripts.extract_feats import init_extract
from scripts.train import init_train

def main():
    args = parse_args()

    if args.train:
        launch_job(args=args, init_method=args.init_method, func=init_train)
    elif args.extract_feats:
        launch_job(args=args, init_method=args.init_method, func=init_extract)
    else:
        print("No script specified, please use [--train, --validate, --extract_feats]")


if __name__ == "__main__":
    main()
