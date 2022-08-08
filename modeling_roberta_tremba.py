import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

class RobertaSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, prompt_length, fine_tune):
        super(RobertaSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddings = self.roberta.get_input_embeddings()

        # ONLY fix roberta
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.cloze_length = prompt_length
        self.pseudo_token_id = self.config.vocab_size
        self.pad_token_id = 1# self.tokenizer.pad_token_id

        self.init_weights()

    def get_query(self, input_id):
        input = input_id.tolist()
        prompt = [self.pseudo_token_id]*self.cloze_length
        query = prompt + input
        return query

    def embed_input(self, queries, prompts):
        # queries [bz, 6, 768]
        if prompts != None:
            assert len(prompts[0]) == self.cloze_length
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1
        raw_embeds = self.embeddings(queries_for_embedding)
        raw_embeds[:, :self.cloze_length, :] = prompts
        return raw_embeds

    def forward(
            self,
            prompts,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if prompts != None:
            bz = len(input_ids)
            queries = []
            for i in range(bz):
                query = self.get_query(input_ids[i])
                queries.append(torch.LongTensor(query).squeeze(0))
            queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
            attention_mask = queries != self.pad_token_id
            inputs_embeds = self.embed_input(queries, prompts)# [bz, 6, 768]
        else:
            inputs_embeds = self.embeddings(input_ids.to(self.device))

        outputs = self.roberta(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if prompts != None:
            sequence_output = outputs[0][:, self.cloze_length:, :]
        else:
            sequence_output = outputs[0]

        if output_attentions == None:
            return sequence_output
        else:
            return (sequence_output,) + (outputs.attentions,)