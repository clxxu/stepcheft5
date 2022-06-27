from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
import torch

def get_outputs(input, prefix, model_paths):

    # set parameters
    learning_rate = 3e-4
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=learning_rate)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = "cpu"

    input = prefix + input

    decode_outputs = []
    for path in model_paths:
        # get model
        checkpoint = torch.load("./Models/" + path, map_location=torch.device('cpu')) 
        model.load_state_dict(checkpoint['model_state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # encode and decode output
        encoding = tokenizer(input, return_tensors="pt").to(device)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        outputs = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), max_length=512)
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = output.replace("u00b0", "\u00B0")
        decode_outputs.append(output)

    return decode_outputs

