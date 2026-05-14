# Joke Generator

A Python-based joke generator that trains a small transformer language model on a dataset of jokes, then lets users generate new jokes from custom prompts.

## What This Project Does

This project has two main parts:

1. **Train the joke model** using `Joke_Gen.py`
2. **Generate jokes** using `prompt.py`

The training script reads jokes from a CSV file named `question_jokes.csv`, tokenizes the text with the GPT-2 tokenizer, trains a transformer-style language model with PyTorch, and saves the trained model as `joke_model.pt`.

After training, `prompt.py` loads `joke_model.pt` and lets the user type prompts to generate new joke text.

## Project Files

```text
.
├── Joke_Gen.py          # Trains the joke-generation model
├── prompt.py            # Runs the trained model and generates jokes
├── requirements.txt     # Python dependencies
├── question_jokes.csv   # Joke dataset, required for training
├── checkpoint.pt        # Training checkpoint, created automatically
├── joke_model.pt        # Final trained model, created after training
└── generated_jokes.txt  # Optional output file for saved generated jokes
```

> **Note:** `question_jokes.csv` is required for training, but it was not included in the uploaded files. Make sure it is in the same folder as `Joke_Gen.py` before training.

## Requirements

This project uses Python and several packages, including:

- `torch`
- `pandas`
- `tiktoken`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## How to Set Up the Project

1. Download or clone the project.
2. Open a terminal in the project folder.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```


## How to Train the Model

Run:

```bash
python Joke_Gen.py
```

The script will:

- Load `question_jokes.csv`
- Convert all jokes to lowercase
- Tokenize the text using the GPT-2 tokenizer
- Train the model for up to 5,000 iterations
- Save progress to `checkpoint.pt`
- Save the final trained model to `joke_model.pt`

During training, the program prints whether it is using CPU or GPU. If CUDA is available, it will use the GPU automatically.

## Training Checkpoints

The training script saves checkpoints automatically.

If `checkpoint.pt` already exists, training will resume from the saved checkpoint instead of starting over.

If you stop training early with `Ctrl + C`, the script saves the current checkpoint before exiting.

## How to Generate Jokes

After training is complete, run:

```bash
python prompt.py
```

The program loads `joke_model.pt` and starts an interactive prompt.

You can enter a starting phrase, such as:

```text
why did the chicken
```

Then the program asks for generation settings:

```text
Max new tokens [200]:
Temperature [0.8]:
Top-k [40, blank for none]:
```

You can press **Enter** to use the default values.

## Generation Settings

### Prompt

The starting text used to begin the joke.

If you leave the prompt blank, the program uses:

```text
why did
```

### Max New Tokens

Controls how much text the model generates.

Default:

```text
200
```

Higher values create longer outputs.

### Temperature

Controls randomness.

Default:

```text
0.8
```

Lower values make the output more predictable. Higher values make the output more random.

### Top-k

Limits the model to choosing from the top possible next tokens.

Default:

```text
40
```

Leaving this blank disables top-k filtering.

## Saving Generated Jokes

After generating text, the program asks:

```text
Save output to generated_jokes.txt? [y/N]:
```

Type:

```text
y
```

if you want to save the generated joke. The output will be written to:

```text
generated_jokes.txt
```

## Example Usage

```bash
python prompt.py
```

Example interaction:

```text
Enter a prompt (or type 'quit' to exit): why did the programmer
Max new tokens [200]:
Temperature [0.8]:
Top-k [40, blank for none]:
```

The program will then print the generated joke text.

To exit the program, type:

```text
quit
```

or:

```text
exit
```

## Important Notes

- You must train the model before running `prompt.py`.
- `prompt.py` requires `joke_model.pt` to exist.
- `Joke_Gen.py` requires `question_jokes.csv` to exist.
- Training can take a while, especially on CPU.
- A GPU will make training faster if CUDA is available.

## Troubleshooting

### `FileNotFoundError: question_jokes.csv`

Make sure `question_jokes.csv` is in the same folder as `Joke_Gen.py`.

### `FileNotFoundError: joke_model.pt`

You need to train the model first:

```bash
python Joke_Gen.py
```

Then run:

```bash
python prompt.py
```

### Invalid number entered

If you type something that is not a number for max tokens, temperature, or top-k, the program will use default values.

## Credits

This project uses PyTorch to build and train the language model and `tiktoken` for GPT-2 tokenization.
