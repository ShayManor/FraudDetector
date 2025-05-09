# 80, 20 train test split
# 50, 50 fraud to not for finetuning
import random

with open('full_data.csv', 'r') as f:
    lines = f.readlines()
ft = ['step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n']
test, train = ['step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n'], ['step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n']
add = False
true, false = ['step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n'], ['step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n']
for line in lines[1:]:
    if line.split(',')[-2].count('1') > 0:
        true.append(line)
        if random.random() <= 0.8:
            ft.append(line)
            add = True
    elif add:
        ft.append(line)
        add = False
    if line.split(',')[-2].count('1') <= 0:
        false.append(line)
    if random.random() < 0.2:
        test.append(line)
    else:
        train.append(line)
with open('finetune.csv', 'w') as f:
    f.writelines(ft)
with open('train.csv', 'w') as f:
    f.writelines(train)
with open('test.csv', 'w') as f:
    f.writelines(test)
with open('true.csv', 'w') as f:
    f.writelines(true)
with open('false.csv', 'w') as f:
    f.writelines(false)