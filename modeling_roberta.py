import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaClassificationHead, RobertaLMHead

class RobertaSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, fine_tune, prompting, prompt_length):
        super(RobertaSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.embeddings = self.roberta.get_input_embeddings()

        # ONLY fix roberta
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.prompting = prompting
        self.pseudo_token_id = self.config.vocab_size
        self.cloze_length = prompt_length
        self.pad_token_id = 1# self.tokenizer.pad_token_id
        if self.prompting:
            self.seq_indices = torch.LongTensor(list(range(self.cloze_length)))
            self.prompt_embedding = torch.nn.Embedding(self.cloze_length, self.config.hidden_size)

        self.init_weights()

    def get_query(self, input_id):
        input = input_id.tolist()
        prompt = [self.pseudo_token_id] * self.cloze_length
        query = prompt + input
        return query

    def embed_input(self, queries, prompts):
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1
        raw_embeds = self.embeddings(queries_for_embedding)
        if prompts == None:
            replace_embeds = self.prompt_embedding(self.seq_indices.to(self.device))
            raw_embeds[:, :self.cloze_length, :] = replace_embeds
        else:
            assert self.cloze_length == len(prompts[0])
            raw_embeds[:, :self.cloze_length, :] = prompts
        return raw_embeds

    def forward(
            self,
            prompts=None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.prompting or prompts != None:
            bz = len(input_ids)
            queries = []
            for i in range(bz):
                query = self.get_query(input_ids[i])
                queries.append(torch.LongTensor(query).squeeze(0))
            queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
            attention_mask = queries != self.pad_token_id
            inputs_embeds = self.embed_input(queries, prompts)
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

            sequence_output = outputs[0][:, self.cloze_length:, :]

        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]

        logits = self.classifier(sequence_output)# [bz, num_labels]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if output_attentions == None:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            output = (logits,) + (outputs.attentions,)
            return ((loss,) + output) if loss is not None else output