# Decisioni fase2.4 — in corso

- Utente vuole: ogni cambio architetturale presentato con domanda su sua opinione prima di applicare.
- Approvato: T_pre4 + T_post4, totale8.
- Poi approvato: corrente post = corrente token + feedback; scelta consigliata dopo ricerca.
- Integrazione avviata dopo conferma:4+4, corrente token mantenuta; nessun altro cambio architetturale.

## Riferimenti per input + contesto

- [Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/html/1409.0473v7): decoder ricorrente condizionato da stato precedente, token precedente e contesto d'attenzione. Precedente per combinare input e contesto, non prova corrente persistente nei nostri LIF.
- [Alayrac et al., Flamingo](https://arxiv.org/html/2204.14198v2): output cross-attention sommato alla rappresentazione via gate tanh, init0, LM preaddestrato congelato. Sostiene fusione additiva modulata; differisce da nostro gate lineare0.1, training da zero, reiniezione nel grafo.
- [Zhou et al., Spikformer](https://arxiv.org/abs/2209.15425): precedente per SNN + self-attention; architettura e attenzione diverse da nostro nucleo anatomico ricorrente con SDPA softmax FP32.

Scelta approvata: mantenere I_token anche nei4 sottopassi post. Motivo: segnale corrente disponibile mentre grafo integra contesto. Rischio: più drive/firing, saturazione o minor uso memoria. Verificare empiricamente; nessun paper prova superiorità di questa variante sul nostro connettoma.

Alternativa solo feedback: stato grafo conserva info sul token elaborato nei4 sottopassi pre. Non equivale a cancellare token.

Test equo: stesso T8, grafo, dati, init; confrontare perdita validation, firing, uso/gradienti attenzione. Distinguere effetto contesto da semplice aumento corrente.
