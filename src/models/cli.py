import argparse
from src.models.llm.nep_gpt import NepaliGPT

def main():
    parser = argparse.ArgumentParser(description="NepaliLekhak Model CLI")
    parser.add_argument("--summary", action="store_true", help="Print model summary")
    args = parser.parse_args()
    if args.summary:
        NepaliGPT.print_model_summary()

if __name__ == "__main__":
    main()