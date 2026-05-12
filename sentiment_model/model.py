from transformers import AutoModelForSequenceClassification, AutoTokenizer

#fix hard coded value to be read from configs
MODEL_NAME = "bert-base-uncased"

def load_model_and_tokenizer(device,model_name="bert-base-uncased",num_classes=2):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer
