import os
from transformers import T5ForConditionalGeneration, T5Tokenizer


def fix_sentence(text: str) -> str:
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "my_t5_fix_sentence_model")

    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    input_text = "fix sentence: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,             
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# -------------------  Uncomment if want to implement the model  ---------------------

# import os
# import pandas as pd
# from datasets import Dataset
# from transformers import TrainingArguments, Trainer
# from transformers import T5ForConditionalGeneration, T5Tokenizer


# # Data source
# # https://huggingface.co/ayush7480/punctuation-restoration-dataset/tree/main
# # Devide this datset --> English value Column as input.txt, Sinhala value column as output.txt 

# BASE_DIR = os.path.dirname(__file__)  # path to Backend folder
# INPUT_TXT = os.path.join(BASE_DIR, "input.txt")
# OUTPUT_TXT = os.path.join(BASE_DIR, "output.txt")

# with open(INPUT_TXT, "r", encoding="utf-8") as f_in:
#     input_lines = f_in.read().strip().splitlines()

# with open(OUTPUT_TXT, "r", encoding="utf-8") as f_out:
#     output_lines = f_out.read().strip().splitlines()

# assert len(input_lines) == len(output_lines), "Line counts do not match!"


# df = pd.DataFrame({
#     "input_text": input_lines,    
#     "target_text": output_lines
# })

# df = df.dropna(subset=["input_text", "target_text"])
# df = df[df["input_text"].str.strip() != ""]
# df = df[df["target_text"].str.strip() != ""]

# df.to_csv("punctuation_segmentation_data.csv", index=False)

# dataset = Dataset.from_pandas(df)

# tokenizer = T5Tokenizer.from_pretrained("t5-small")

# dataset = Dataset.from_pandas(df)

# dataset = dataset.shuffle(seed=42).select(range(min(5000, len(dataset))))

# def preprocess(example):
#     input_text = "fix sentence: " + example["input_text"]
#     target_text = example["target_text"]

#     input_enc = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
#     target_enc = tokenizer(target_text, truncation=True, padding="max_length", max_length=128)

#     input_enc["labels"] = target_enc["input_ids"]
#     return input_enc

# tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# split = tokenized_dataset.train_test_split(test_size=0.1)
# train_data = split["train"]
# eval_data = split["test"]


# training_args = TrainingArguments(
#     output_dir="./t5_fix_sentence_model",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=20,
#     weight_decay=0.01,
#     logging_steps=10,
#     save_total_limit=1
# )

# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     tokenizer=tokenizer
# )

# trainer.train()
# trainer.save_model("my_t5_fix_sentence_model")

# # Load trained model
# model = T5ForConditionalGeneration.from_pretrained("my_t5_fix_sentence_model")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")


# def fix_sentence(text):
#     input_text = "fix sentence: " + text
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
#     output_ids = model.generate(
#         **inputs,
#         max_length=128,
#         num_beams=4,             
#         early_stopping=True
#     )
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# print(fix_sentence("nomatterathowmanyplacesyoulookforhappinessyouwon'tfinditbecauseyouneverlostitoutsideitsstillinsideofyou"))

