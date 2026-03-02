import torch
from einops import rearrange
import pytorch_lightning as pl
from Mesh_Preparation.shapenetcorev2_prep_fns import map_shapeNetCorev2
from typing import Any

class CLS(pl.LightningModule):
    def __init__(self, dim_size: int, num_classes: int):
        super(CLS, self).__init__()

        self.dim_size = dim_size
        self.num_classes = num_classes

        self.criterion = torch.nn.CrossEntropyLoss()
        self.classifier = torch.nn.Linear(self.dim_size, self.num_classes)

    def classification_loss(self,  predicted_cls_token: torch.Tensor, gt_cls_token: torch.Tensor):
        # target = torch.empty(batch_size, dtype=torch.long, device=self.device).random_(self.num_classes)
        # print("Target:", target, ", shape: ", target.shape)
        assert gt_cls_token.shape == predicted_cls_token.shape
        classification_loss = self.criterion(predicted_cls_token, gt_cls_token)
        return classification_loss

    def incorporate_cls_token(self, input_sequence: torch.Tensor, cls_tokens: torch.Tensor, penc_cls: torch.Tensor) -> torch.Tensor:
        B, SeqL, Ch = input_sequence.shape
        lt_dim, D, H, W = cls_tokens.shape
        cls_tokens_reshaped = rearrange(cls_tokens, " lt_dim D H W -> (lt_dim D H W) 1")
        # cls token [1, 1, 1024] ==> broadcast/tile to [B,1,1024] ==> concatenate / hstack to existing sequence
        # concatenate PE with the cls_token :
        cls_token_plus_pe = torch.cat((penc_cls, cls_tokens_reshaped), dim=0)
        cls_token_plus_pe_re = rearrange(cls_token_plus_pe, "A B-> 1 B A")
        input_sequence_plus_cls = torch.cat((torch.tile(cls_token_plus_pe_re, (B, 1, 1)), input_sequence.clone()), dim=1)  # the output shape will be  [B, 1+SeqL, Ch]
        assert input_sequence_plus_cls.shape == (B, SeqL + 1, Ch)
        return input_sequence_plus_cls

    def extract_cls_token_from_output(self, output_sequence: torch.Tensor) -> torch.Tensor:

        cls_token_learned = output_sequence[:, 0, :]
        return cls_token_learned

    def make_one_hot_vector(self, labels: tuple):
        batch_size = len(labels)
        one_hot_vector = torch.zeros((batch_size, self.num_classes), dtype=torch.float32, device='cuda')
        # class_ids = torch.zeros([batch_size], dtype=torch.int64)
        for b in range(batch_size):
            class_id = map_shapeNetCorev2(labels[b])
            one_hot_vector[b, class_id] = 1
        #     class_ids[b] = class_id
        # class_ids_one_hot = torch.nn.functional.one_hot(class_ids, self.num_classes)
        # one_hot_vector_max = one_hot_vector.argmax(dim=1, keepdim=True)
        return one_hot_vector

    def forward(self, output_sequence: torch.Tensor) -> torch.Tensor:
        extracted_cls_token = self.extract_cls_token_from_output(output_sequence.clone())
        predicted_cls_token = self.classifier(extracted_cls_token)

        return predicted_cls_token
