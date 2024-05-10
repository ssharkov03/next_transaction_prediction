import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, GPT2Model, GPT2Config, MambaConfig, MambaModel, JambaConfig, JambaModel


class LSTMSequenceModel(nn.Module):
    def __init__(self, hidden_dim, feature2vocab_dict):
        super(LSTMSequenceModel, self).__init__()
        self.features_list = feature2vocab_dict.keys()
        self.embeddings = nn.ModuleList([nn.Embedding(feature2vocab_dict[feature_name], hidden_dim) for feature_name in self.features_list])
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature2vocab_dict['mcc'])  # projection to dict

    def forward(self, x):  # batch_size x seq_len x num_features
        embedded = torch.cat([self.embeddings[i](x[..., i]).unsqueeze(2) for i in range(len(self.features_list))], dim=2)  # batch_size x seq_len x num_features x hidden_dim
        
        # суммируем вдоль фичей
        embedded = embedded.sum(dim=2)  # batch_size x seq_len x hidden_dim
        
        # LSTM over the sequence
        output, (hn, cn) = self.lstm(embedded)
        
        # Final output from the last hidden state
        out = self.fc(output)  # batch_size x seq_len x vocab_size
        return out


class BertSequenceModel(nn.Module):
    def __init__(self, hidden_dim, feature2vocab_dict):
        super(BertSequenceModel, self).__init__()
        self.features_list = feature2vocab_dict.keys()
        config = BertConfig(hidden_size=hidden_dim, num_hidden_layers=4, num_attention_heads=4) 
        self.bert = BertModel(config)
        self.embeddings = nn.ModuleList([nn.Embedding(feature2vocab_dict[feature_name], hidden_dim) for feature_name in self.features_list])
        self.output_layer = nn.Linear(hidden_dim, feature2vocab_dict['mcc'])  # projection to dict 
        
    def forward(self, x):  # batch_size x seq_len x num_features
        x = [self.embeddings[i](x[..., i]).unsqueeze(2) for i in range(len(self.features_list))] 
        x = torch.cat(x, dim=2) # batch_size x seq_len x num_features x hidden_dim

        # Суммируем вдоль фичей
        x = x.sum(dim=2)  # batch_size x seq_len x hidden_dim
        
        # Создаем маску для аттеншна, чтобы иммитировать авторегрессивность (не заглядываем вперед)
        batch_size = x.size(0)
        seq_length = x.size(1)
        causal_mask = torch.tril(torch.ones((batch_size, seq_length, seq_length), dtype=torch.long)).to(x.device)

        # Bert
        outputs = self.bert(inputs_embeds=x, attention_mask=causal_mask)        
        last_hidden_states = outputs.last_hidden_state

        # Apply the output layer to predict the next feature
        logits = self.output_layer(last_hidden_states)  # batch_size x seq_len x vocab_size
        return logits


class GPT2SequenceModel(nn.Module):
    def __init__(self, hidden_dim, feature2vocab_dict):
        super(GPT2SequenceModel, self).__init__()
        self.features_list = feature2vocab_dict.keys()
        self.embeddings = nn.ModuleList([nn.Embedding(feature2vocab_dict[feature_name], hidden_dim) for feature_name in self.features_list])
        config = GPT2Config(n_embd=hidden_dim, n_layer=4, n_head=4)
        self.gpt2 = GPT2Model(config)
        self.output_layer = nn.Linear(hidden_dim, feature2vocab_dict['mcc']) # projection to dict
        
    def forward(self, x):  # batch_size x seq_len x num_features
        x = [self.embeddings[i](x[..., i]).unsqueeze(2) for i in range(len(self.features_list))]
        x = torch.cat(x, dim=2) # batch_size x seq_len x num_features x hidden_dim

        # Суммируем вдоль фичей
        x = x.sum(dim=2)  # batch_size x seq_len x hidden_dim

        outputs = self.gpt2(inputs_embeds=x)
        last_hidden_states = outputs.last_hidden_state

        # Apply the output layer to predict the next feature
        logits = self.output_layer(last_hidden_states)  # batch_size x seq_len x vocab_size
        return logits


class SelectiveStateSpace(nn.Module):
    """
    Класс необходимый для mamba
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelectiveStateSpace, self).__init__()
        # Define parameters for the state space transformation
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Dynamic parameters adjusted per input
        self.A = nn.Linear(input_dim, hidden_dim * hidden_dim)
        self.B = nn.Linear(input_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, output_dim)

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters for stability
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.C.weight, a=math.sqrt(5))

    def forward(self, x, h_prev):
        # x is batch_size x input_dim
        # h_prev is batch_size x hidden_dim
        # Compute parameter matrices from input
        A_dyn = self.A(x).view(-1, self.hidden_dim, self.hidden_dim)
        B_dyn = self.B(x)
        C_dyn = self.C(h_prev)

        # State update equation h(t) = A * h(t-1) + B * x(t)
        h_next = torch.bmm(A_dyn, h_prev.unsqueeze(2)).squeeze(2) + B_dyn

        # Output equation y(t) = C * h(t)
        y = C_dyn
        return h_next, y


# class MambaSequenceModel(nn.Module):
#     def __init__(self, hidden_dim, feature2vocab_dict):
#         super(MambaSequenceModel, self).__init__()
#         self.features_list = feature2vocab_dict.keys()
#         self.embeddings = nn.ModuleList([nn.Embedding(feature2vocab_dict[feature_name], hidden_dim) for feature_name in self.features_list])
#         output_vocab_size = feature2vocab_dict['mcc']
#         self.hidden_dim = hidden_dim
#         self.ssm = SelectiveStateSpace(hidden_dim, hidden_dim, output_vocab_size)

#     def forward(self, x):  # batch_size x seq_len x num_features
#         embedded = torch.cat([self.embeddings[i](x[..., i]).unsqueeze(2) for i in range(len(self.features_list))], dim=2) # batch_size x seq_len x num_features x hidden_dim
        
#         # Суммируем вдоль фичей
#         embedded = embedded.sum(dim=2)  # batch_size x seq_len x hidden_dim

#         # Mamba magic
#         batch_size, seq_len, _ = embedded.size()
#         h = torch.zeros(batch_size, self.hidden_dim, device=embedded.device)

#         outputs = []
#         for t in range(seq_len):
#             h, y = self.ssm(embedded[:, t, :], h)
#             outputs.append(y)

#         logits = torch.stack(outputs, dim=1)
#         return logits


class MambaSequenceModel(nn.Module):
    def __init__(self, hidden_dim, feature2vocab_dict):
        super(MambaSequenceModel, self).__init__()
        self.features_list = feature2vocab_dict.keys()
        self.embeddings = nn.ModuleList([nn.Embedding(feature2vocab_dict[feature_name], hidden_dim) for feature_name in self.features_list])
        config = MambaConfig(hidden_size=hidden_dim, num_hidden_layers=4, state_size=8)
        self.mamba = MambaModel(config)
        self.output_layer = nn.Linear(hidden_dim, feature2vocab_dict['mcc']) # projection to dict


    def forward(self, x):  # batch_size x seq_len x num_features
        embedded = torch.cat([self.embeddings[i](x[..., i]).unsqueeze(2) for i in range(len(self.features_list))], dim=2) # batch_size x seq_len x num_features x hidden_dim
        
        # Суммируем вдоль фичей
        embedded = embedded.sum(dim=2)  # batch_size x seq_len x hidden_dim

        outputs = self.mamba(inputs_embeds=embedded)
        last_hidden_states = outputs.last_hidden_state

        # Apply the output layer to predict the next feature
        logits = self.output_layer(last_hidden_states)  # batch_size x seq_len x vocab_size
        return logits


class JambaSequenceModel(nn.Module):
    def __init__(self, hidden_dim, feature2vocab_dict):
        super(JambaSequenceModel, self).__init__()
        self.features_list = feature2vocab_dict.keys()
        self.embeddings = nn.ModuleList([nn.Embedding(feature2vocab_dict[feature_name], hidden_dim) for feature_name in self.features_list])
        config = JambaConfig(hidden_size=hidden_dim, num_hidden_layers=4, intermediate_size=hidden_dim * 2, num_key_value_heads=3, num_attention_heads=4, num_experts=4, state_size=8, use_mamba_kernels=False)
        self.jamba = JambaModel(config)
        self.output_layer = nn.Linear(hidden_dim, feature2vocab_dict['mcc']) # projection to dict


    def forward(self, x):  # batch_size x seq_len x num_features
        embedded = torch.cat([self.embeddings[i](x[..., i]).unsqueeze(2) for i in range(len(self.features_list))], dim=2) # batch_size x seq_len x num_features x hidden_dim
        
        # Суммируем вдоль фичей
        embedded = embedded.sum(dim=2)  # batch_size x seq_len x hidden_dim

        # Создаем маску для аттеншна, чтобы иммитировать авторегрессивность (не заглядываем вперед)
        batch_size = embedded.size(0)
        seq_length = embedded.size(1)
        causal_mask = torch.tril(torch.ones((batch_size, seq_length, seq_length), dtype=torch.long)).to(x.device)


        outputs = self.jamba(inputs_embeds=embedded, attention_mask=causal_mask)
        last_hidden_states = outputs.last_hidden_state

        # Apply the output layer to predict the next feature
        logits = self.output_layer(last_hidden_states)  # batch_size x seq_len x vocab_size
        return logits



if __name__ == '__main__':
    from dataset import get_mappings
    from utils import get_device
    _, feature2vocab_size = get_mappings(train_data="../data/part_000_0_to_23646.parquet", features_list=["mcc", "day_of_week", "payment_system"])


    # Простой конфиг модели
    hidden_dim = 128 
    device = get_device()

    model_classes = [LSTMSequenceModel, BertSequenceModel, GPT2SequenceModel, MambaSequenceModel, JambaSequenceModel]
    for model_class in model_classes:
        model = model_class(hidden_dim, feature2vocab_dict=feature2vocab_size)
        model.to(device)

        input_tensor = torch.randint(0, 1, (10, 7, 3))  # (batch_size, seq_len, num_features) (10, 7, 3)
        print(f"\nTesting {str(model_class)}\nInput tensor shape:", input_tensor.shape)
        logits = model(input_tensor.to(device))
        print("Output tensor shape:", logits.shape)  # (batch_size, seq_len, target_feature_vocab_size) (10, 7, 112)  
