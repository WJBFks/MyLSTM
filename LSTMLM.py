# %%
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test

# 优先选择设备(cuda | cpu)
dev_first = "cuda:0"
# 选择模型('1':单层模型, 'n':多层模型)
MyLSTM_select = 'n'
# LSTM多层模型层数(当选择多层模型时有效)
MyLSTM_layer = 1

device = torch.device(dev_first if torch.cuda.is_available() else "cpu")

'''
单层LSTM
'''


class MyLSTM1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTM1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input_gate
        self.i_i = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.i_h = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # forget_gate
        self.f_i = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.f_h = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # cell
        self.g_i = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.g_h = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # output_gate
        self.o_i = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.o_h = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, X):
        ct = torch.zeros(self.hidden_size, self.hidden_size).to(device)
        ht = torch.zeros(self.hidden_size, self.hidden_size).to(device)
        outputs = []
        for x in X:
            it = torch.sigmoid(self.i_i(x) + self.i_h(ht))
            ft = torch.sigmoid(self.f_i(x) + self.f_h(ht))
            gt = torch.tanh(self.g_i(x) + self.g_h(ht))
            ot = torch.sigmoid(self.o_i(x) + self.o_h(ht))
            ct = ft * ct + it * gt
            ht = ot * torch.tanh(ct)
            outputs.append(ht)
        return outputs


'''
多层LSTM
'''


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer=1):
        super(MyLSTM, self).__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input_gate
        self.i_i = [nn.Linear(self.input_size if l == 0 else self.hidden_size, self.hidden_size, bias=True).to(device)
                    for l in range(self.layer)]
        self.i_h = [nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
                    for _ in range(self.layer)]
        # forget_gate
        self.f_i = [nn.Linear(self.input_size if l == 0 else self.hidden_size, self.hidden_size, bias=True).to(device)
                    for l in range(self.layer)]
        self.f_h = [nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
                    for _ in range(self.layer)]
        # cell
        self.g_i = [nn.Linear(self.input_size if l == 0 else self.hidden_size, self.hidden_size, bias=True).to(device)
                    for l in range(self.layer)]
        self.g_h = [nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
                    for _ in range(self.layer)]
        # output_gate
        self.o_i = [nn.Linear(self.input_size if l == 0 else self.hidden_size, self.hidden_size, bias=True).to(device)
                    for l in range(self.layer)]
        self.o_h = [nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
                    for _ in range(self.layer)]

    def forward(self, X):
        ct = [torch.zeros(self.hidden_size, self.hidden_size).to(device)
              for _ in range(self.layer)]
        ht = [torch.zeros(self.hidden_size, self.hidden_size).to(device)
              for _ in range(self.layer)]
        outputs = []
        for x in X:
            for l in range(self.layer):
                it = torch.sigmoid(self.i_i[l](x) + self.i_h[l](ht[l]))
                ft = torch.sigmoid(self.f_i[l](x) + self.f_h[l](ht[l]))
                gt = torch.tanh(self.g_i[l](x) + self.g_h[l](ht[l]))
                ot = torch.sigmoid(self.o_i[l](x) + self.o_h[l](ht[l]))
                ct[l] = ft * ct[l] + it * gt
                ht[l] = ot * torch.tanh(ct[l])
                x = ht[l]
            outputs.append(ht[l])
        return outputs


'''
模型
'''


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        if MyLSTM_select == '1':    # 单层模型
            self.LSTM = MyLSTM1(emb_size, n_hidden)
        else:                       # 多层模型
            self.LSTM = MyLSTM(emb_size, n_hidden, MyLSTM_layer)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)
        outputs = self.LSTM(X)
        outputs = outputs[-1]
        model = self.W(outputs) + self.b
        return model


def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            # create (1~n-1) as input
            input = [word2number_dict[n]
                     for n in word[word_index:word_index+n_step]]
            # create (n) as target, We usually call this 'casual language model'
            target = word2number_dict[word[word_index+n_step]]
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    # (batch num, batch size, n_step) (batch num, batch size)
    return all_input_batch, all_target_batch


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict


def train_LSTMlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(
            data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(
            all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        # valid and test batch size is 128
        total_valid = len(all_valid_target)*128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch+1}.ckpt')


def test_LSTMlm(select_model_path):
    # load the selected model
    model = torch.load(
        select_model_path, map_location=(dev_first if torch.cuda.is_available() else "cpu"))
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(
        data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(
        all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target)*128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell
    batch_size = 128  # batch size
    learn_rate = 0.0005
    all_epoch = 5  # the all epoch for training
    emb_size = 256  # embeding size
    # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    save_checkpoint_epoch = 5
    data_root = 'penn_small'
    # the path of train dataset
    train_path = os.path.join(data_root, 'train.txt')

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    # print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(
        train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(
        all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)

# %%
