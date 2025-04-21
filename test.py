import torch
from torch import nn, sigmoid
from torch.utils.data import DataLoader

from train import FraudDataset, FraudClassifier


def check_correct(logit, label):
    prob = sigmoid(logit).item()
    pred_class = 1 if prob >= 0.5 else 0
    return pred_class == int(label.item())

def test(weights):
    dataset = FraudDataset('test.csv')
    dataloader = DataLoader(batch_size=6, dataset=dataset, shuffle=True)
    model = FraudClassifier()
    state = torch.load(weights, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    correct_counter, total = 0, 0
    test_loss = 0

    for data, label in dataloader:
        output = model(data)
        pred = output.data
        print(f"Pred: {pred}")
        correct = check_correct(pred, label)
        if not correct:
            print(f'Incorrect case. Expected: {label[0]}, Received: {pred[0]}')
        correct_counter += int(correct)
        total += label.size(0)
        loss = criterion(output, label)
        test_loss += loss.item() * data.size(0)
    print(f'Testing Loss:{test_loss / len(dataloader)}')
    print(f'Correct Predictions: {correct_counter}/{total}')
test('weights.pt')