import torch

from data_preprocess import load_imdb
from model import BiLSTM
from torchtext.data import get_tokenizer


def predict_sentiment(text, model, tokenizer, vocab, device):
    # Tokenize the input text
    tokens = tokenizer(text)
    indexed_tokens = [vocab[token] for token in tokens]
    tensor_tokens = torch.LongTensor(indexed_tokens).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        model.eval()
        output = model(tensor_tokens)
        probabilities = torch.softmax(output, dim=1)

    # Predict sentiment
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment = "positive" if predicted_class == 1 else "negative"
    confidence = probabilities.squeeze(0)[predicted_class].item()

    return sentiment, confidence


def save_vocab_to_txt(vocab, file_path):
    sorted_vocab = sorted(vocab.get_stoi().items(), key=lambda x: x[1])
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, index in sorted_vocab:
            f.write(f"{word}    {index}\n")


if __name__ == "__main__":
    # Load the saved model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, vocab = load_imdb()
    model = BiLSTM(vocab).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    # Tokenizer
    tokenizer = get_tokenizer('basic_english')
    # Input text for inference
    text_to_predict = "This movie was fantastic! I loved every moment of it."
    # text_to_predict = "The service was slow, the food was bland and overpriced, and the atmosphere was unpleasant."

    # Perform inference
    sentiment, confidence = predict_sentiment(text_to_predict, model, tokenizer, vocab, device)
    print(f"Predicted sentiment: {sentiment}, Confidence: {confidence:.4f}")
