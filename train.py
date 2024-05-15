import random
import torch
import numpy as np
from data_preprocess import load_imdb
from torch.utils.data import DataLoader
from model import BiLSTM
import torch.nn as nn


def main():
    set_seed() # 使用随机数 确保在使用随机数时能够获得可重复的结果

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10

    train_data, test_data, vocab = load_imdb()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BiLSTM(vocab).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}\n' + '-' * 32)
        avg_train_loss = 0
        model.train()  # 将模型设置为训练模式
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            avg_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 5 == 0:
                print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss.item():.4f}")

        print(f"Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n")

        acc = 0
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                acc += (pred.argmax(1) == y).sum().item()

        accuracy = acc / len(test_loader.dataset)
        print(f"Accuracy: {accuracy:.4f}")

        # Save the model if it has the highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

    print("Training completed!")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
