import torch
from torch import nn, sigmoid
from torch.utils.data import DataLoader

from FraudDataset import FraudDataset
from FraudClassifier import FraudClassifier


def check_correct(logit, label):
    prob = sigmoid(logit[0]).item()
    pred_class = 1 if prob >= 0.5 else 0
    return pred_class == int(label[0].item())


def test(weights, csv_path):
    dataset = FraudDataset(csv_path)
    dataloader = DataLoader(batch_size=1, dataset=dataset, shuffle=True)
    model = FraudClassifier()
    state = torch.load(weights)
    model.load_state_dict(state)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    correct_counter, total = 0, 0
    test_loss = 0

    for data, label in dataloader:
        output = model(data)
        pred = output.data
        correct = check_correct(pred, label)
        if not correct:
            probs = torch.sigmoid(pred)
            exp = torch.sigmoid(label)
            # print(f'Incorrect case. Expected: {exp}, Received: {probs}')
        correct_counter += int(correct)
        total += label.size(0)
        loss = criterion(output.flatten(), label.flatten())
        test_loss += loss.item() * data.size(0)
    print(f'Testing Loss:{test_loss / len(dataloader)}')
    print(f'Correct Predictions: {correct_counter}/{total}')


def full_test():
    print('Begin testing for weights')
    test('weights.pt', 'test.csv')
    test('weights.pt', 'true.csv')
    test('weights.pt', 'false.csv')
    print('---------------------------')
    print('Begin testing for finetuned weights')
    test('ft_weights.pt', 'test.csv')
    test('ft_weights.pt', 'true.csv')
    test('ft_weights.pt', 'false.csv')


if __name__ == '__main__':
    full_test()
