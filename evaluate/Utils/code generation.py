import torch


def generate_code_by_marian(model, tokenizer, input_sentence="I want to buy a car"):
    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids
    decoder_input_ids = tokenizer("<s>", add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
    encoded_sequence = (outputs.encoder_last_hidden_state,)
    # lm_logits = outputs.logits
    # next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
    # decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
    #
    # lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids,
    #                   return_dict=True).logits
    # next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
    # decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)

    lm_logits = model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids,
                      return_dict=True).logits
    next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
    decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)

    output = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

    return output
