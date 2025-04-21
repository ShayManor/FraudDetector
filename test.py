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
    state = torch.load(weights, map_location=torch.device('cpu'))
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
    print('True test:')
    test('weights.pt', 'true.csv')
    print('False test:')
    test('weights.pt', 'false.csv')
    print('---------------------------')
    print('Begin testing for finetuned weights')
    test('ft_weights.pt', 'test.csv')
    print('True test:')
    test('ft_weights.pt', 'true.csv')
    print('False test:')
    test('ft_weights.pt', 'false.csv')


if __name__ == '__main__':
    full_test()

# Begin testing for weights
# Testing Loss:0.006604670058826224
# Correct Predictions: 989669/991178
# True test:
# Testing Loss:90.53193622773803
# Correct Predictions: 30/8213
# False test:
# Testing Loss:0.007720230532787741
# Correct Predictions: 4947013/4955123
# ---------------------------
# Begin testing for finetuned weights
# Testing Loss:0.9638416284434608
# Correct Predictions: 568195/991178
# True test:
# Testing Loss:4.494162518536319
# Correct Predictions: 5423/8213
# False test:
# Testing Loss:0.9736380548753285
# Correct Predictions: 2833015/4955123