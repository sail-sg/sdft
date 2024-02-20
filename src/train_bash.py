from llmtuner import run_exp
import logging

# Disable logging
logging.disable(logging.CRITICAL + 1)

def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
