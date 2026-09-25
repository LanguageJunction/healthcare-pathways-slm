from transformers import AutoTokenizer, PreTrainedTokenizerFast


# MODEL_NAME = "distilbert/distilgpt2"
MODEL_NAME = "alokanand002/medical-bpe-16k"


def get_tokenizer():

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    print(tokenizer.tokenize("acetylcholinesterase inhibitor"))


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def main():

    tokenizer = get_tokenizer()

    print(
        "Vocabulary size:",
        len(tokenizer)
    )

if __name__ == "__main__":
    main()