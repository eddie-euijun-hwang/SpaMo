import h5py
from transformers import AutoTokenizer

POLISH_TO_KNOWN_LETTER = {
    "Ą": "A",
    "Ć": "C",
    "Ę": "E",
    "Ł": "L",
    "Ń": "N",
    "Ó": "O",
    "Ś": "S",
    "Ź": "Z",
    "Ż": "Z",
    "ą": "a",
    "ć": "c",
    "ę": "e",
    "ł": "l",
    "ń": "n",
    "ś": "s",
    "ź": "z",
    "ż": "z",
}

TRANSLATION_TABLE = str.maketrans(POLISH_TO_KNOWN_LETTER)
TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-xl")
UNK_SYMBOL = 2
problematic_sentences = []


def process_sentence(sentence: str):
    return sentence.translate(TRANSLATION_TABLE)


def is_problematic(sentence: str) -> bool:
    tokens = TOKENIZER(sentence).input_ids
    if UNK_SYMBOL in tokens:
        return True

    return False


def main():
    f = h5py.File("features/texts.h5", mode="r+")

    keys = list(f.keys())
    problematic_sentences = []
    for key in keys:
        sentence = f[key][()].decode()
        translated = process_sentence(sentence)

        if is_problematic(translated):
            problematic_sentences.append((translated, key))

        del f[key]
        f.create_dataset(key, data=translated.encode())

    f.close()
    print(problematic_sentences)
    print(len(problematic_sentences))


if __name__ == "__main__":
    main()
