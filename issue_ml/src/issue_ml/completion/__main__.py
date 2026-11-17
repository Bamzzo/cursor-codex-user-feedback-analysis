import argparse
from .experiment import load_config,run

def main():
    parser=argparse.ArgumentParser(description='Random forest Issue completed prediction')
    parser.add_argument('--config',required=True)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    run(load_config(args.config),args.audit_only)

if __name__=='__main__':main()
