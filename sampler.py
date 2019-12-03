import numpy as np
from utils_inference import sentence_from_indices, beam_search_decoder
from alignment_utils import tokenize_mr, tokenize_mr_upper
from data_processing import Delexicalizer
from slot_aligner import SlotAligner

class NMTSampler:
    def __init__(self, vectorizer, model, use_reranker, beam_width=3):
        self.vectorizer = vectorizer
        self.model = model
        self.delexicalizer = Delexicalizer("partial", ["name", "near", "food"])
        self.aligner = SlotAligner()
        self.use_reranker = use_reranker
        self.beam_width = beam_width
    
    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict
        
        y_pred = self.model(x_source=batch_dict['x_source'], 
                            x_source_lengths=batch_dict['x_source_length'], 
                            target_sequence=batch_dict['x_target'],
                            sample_probability=1.0)
        self._last_batch['y_pred'] = y_pred
        attention_batched = np.stack(self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        self._last_batch['attention'] = attention_batched
        
    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)
    
    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch['y_target'][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)
    
    def _get_gt_mrs(self, index):
        inp_mr = self._last_batch['inp_gt'][index]
        return inp_mr
    
    def _get_gt_ref(self, index):
        ref_gt = self._last_batch['ref_gt'][index]
        return ref_gt
    
    def _get_sampled_sentence(self, index, return_string=True):
        vocab = self.vectorizer.target_vocab

        # Convert the scores for the words to softmax probabilities
        prob_res = torch.nn.functional.softmax(self._last_batch['y_pred'], dim=2)
        sentece_probs = prob_res[index].cpu().detach().numpy()
        
        top_sequences = beam_search_decoder(sentece_probs, self.beam_width)

        gt_mr = tokenize_mr(self._last_batch['inp_gt'][index])

        # By default without reranking the index of the max sequence will be 0
        max_seq_idx = 0

        if self.use_reranker:
            res_after_reranking = []
            # Iterate over all produced sequences
            for sequence in top_sequences:

                # Produce the sentence
                sent_str = sentence_from_indices(sequence[0], vocab, return_string=True)

                # Get the reranker score
                reranker_score = self.aligner.alignment_reranker(gt_mr, sent_str)

                # Produce the new sentence probability 
                res_after_reranking.append(sequence[1]*reranker_score)  

                # Get the sentence with max probability of after both beam search and reranking
                max_seq_idx = res_after_reranking.index(max(res_after_reranking))
        
        sentence_indices = top_sequences[max_seq_idx][0]
        
        print("Making a prediction {}".format(index))
        
        return sentence_from_indices(sentence_indices, vocab, return_string=return_string)

    def get_ith_item(self, index, return_string=True):
        sampled_sentence = self._get_sampled_sentence(index, return_string=return_string)
        gt_mrs = self._get_gt_mrs(index)
        ref_gt = self._get_gt_ref(index)
        
        sampled_normalized = self.delexicalizer.reverse_delexicalize_sample(gt_mrs, " ".join(sampled_sentence))
        
        output = {"source": self._get_source_sentence(index, return_string=return_string), 
                  "reference": self._get_reference_sentence(index, return_string=return_string), 
                  "sampled": sampled_sentence,
                  "attention": self._last_batch['attention'][index],
                  "sampled_normalized":sampled_normalized,
                  "reference_gt":ref_gt,
                  "mrs_gt":gt_mrs,
                  "y_target": self._last_batch['y_target'][index]
                 }
        
        reference = output['reference']
        hypothesis = output['sampled']
        
        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)
        
        output['bleu-4'] = bleu_score.sentence_bleu(references=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=chencherry.method1)
        
        return output