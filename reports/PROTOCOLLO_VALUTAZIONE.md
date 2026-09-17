# Protocollo v1 — fissato prima del training LM

Data: 2026-09-12. Fase1; nessun punteggio LM osservato. Budget pianificati; nessun training LM eseguito in fase1.

## Dati / separazione

- TinyStoriesV2-GPT4 locale; confronto SHA256 con LFS upstream. Provenienza in `dataset/prepared_v1/upstream_metadata.json`.
- Train ufficiale deduplicato: SHA256 testo whitespace collassato, confronto case-sensitive. Payload conservato salvo whitespace ASCII esterno.
- Validation ufficiale: primo frammento e coda incompleta esclusi; dedup; hash modulo5 → se0 test, altrimenti development.
- Duplicati train/heldout: tenere heldout, escludere train. Nessuna dedup semantica garantita.
- Test interno, non benchmark ufficiale indipendente. Dataset pubblico: contaminazione modelli preaddestrati non esclusa.
- Tokenizer allenato su campione deterministico train (~1/64 storie); confronto su altro campione train (~1/256), nessun fitting su dati riservati.
- Byte-level BPE4k baseline; PAD0, BOS1, EOS2. Storie complete: `[BOS,payload,EOS]`; padding escluso da loss e stato.
- File uint16 little-endian + offset uint64 + conteggio byte UTF8 uint32; accesso mmap. Manifest hash di fonti, selezioni, tokenizer e output.

## Suite fissata

| Prova | Campione / metrica |
|---|---|
| Modellazione storie | 512 storie test hash normalizzato minore; bit/byte payload; EOS separato |
| Completamento | Prime64 delle512; prefisso prime20 parole (split whitespace); continuazione max80 token; stop EOS |
| Copia/recupero | Task sintetici separati: alfabeto8 simboli, ritardi8/32/64; 256 esempi/ritardo; exact accuracy |
| Diagnosi validation | 128 storie development hash minore; CE/bit-byte; mai test per selezione checkpoint |

- Generazione: greedy primaria; campionamento secondario temperatura0.8, top-p0.9, semi17/29/43. Conservare output integrali e prompt, nessuna scelta estetica post-generazione.
- Qualità descrittiva: rubriche grammatica/coerenza/ripetizione 0–2 ciascuna, confronto cieco su tutte64; nessun giudice LLM obbligatorio. Non usare rubriche come unica prova vittoria.
- Copia/ritardo: specifica operativa task da fissare in fase3 prima dei risultati; train/dev/test sintetici disgiunti, semi171/172/173. Prova diagnostica, non benchmark storico.

## Baseline

- Unigram e bigram add-alpha0.1: stesso tokenizer, stesso train del modello confrontato.
- GRU da zero: embedding256 legato, 2 strati hidden256, proiezione256, dropout0.
- Decoder Transformer da zero: d256, 2 blocchi pre-LayerNorm, MHA4, FFN1024 GELU, RoPE, contesto128, embedding legato, dropout0.
- Riferimento storico: `openai-community/gpt2` (GPT-2 small, 2019), checkpoint congelato e tokenizer nativo. Revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, in `dataset/prepared_v1/historical_baseline.json`; pesi non scaricati, nessun fine-tuning nel confronto preaddestrato.
- GPT-2 ≠ confronto a compute/dati uguali; vincere su storie semplici non batte generalmente. Bit/byte confrontabile con convenzioni comuni; PPL tra tokenizer diversi vietata.
- Connettoma: senza attenzione; core congelato; grafo rimescolato preservando gradi/segni quando fattibile; core sostituito. Riaddestrare controlli, dichiarare parametri e raggiungibilità.

## Budget pilota / selezione

- Fase3: max10 minuti/caso diagnostico dopo caricamento, valutazione incl.; stop immediato per guardie/NaN. Non prova convergenza lunga.
- Fase5 pilota: max6 ore per modello/run, seed17; tetto1.000.000 token target validi. Nessun obbligo consumare budget se instabile.
- Confronto dati uguali: snapshot dopo100.000 target validi, stesso ordine storie/seme.
- Confronto tempo uguale: snapshot a2 ore, warmup/training/validation inclusi; caricamento/preparazione registrati a parte. Nessuno snapshot se tetto dati o stop precede; confronto allora non disponibile.
- Validation ogni30 minuti e a fine run; checkpoint per minima loss validation. Test letto per punteggi solo fase6.
- Budget fase6: max2 ore/modello valutazione pilota; se suite non termina, riportare incompleto, non sostituire campione dopo risultati.
- Estensione/re-seed: nuova versione protocollo prima dei risultati aggiuntivi, motivazione esplicita.

## Convenzioni / criteri

- Bit/byte = NLL payload totale / (ln2 × byte UTF8 payload totale). BOS condizionante non scored; EOS separato. Testi identici, reset a inizio storia.
- Confronti scratch: tokenizer4k comune. Contesto nativo dichiarato (mosca stato ricorrente+cache128; decoder128; GRU ricorrente). GPT-2 finestra nativa max1024 token, scoring rolling senza doppio conteggio; nessuna equivalenza memoria implicita.
- Perdita, accuracy, byte/s, token/s, picchi RAM/VRAM, parametri addestrabili/fissi, archi e sinapsi mantenuti, dati/tempo effettivi.
- Vittoria primaria: bit/byte inferiore sullo stesso test; riportare differenza e intervallo bootstrap appaiato per storia (1000 repliche, seed17). Una seed non prova robustezza generale.
- Completamento e task memoria riportati separatamente; nessun punteggio aggregato inventato per nascondere sconfitte.
- Modello selezionato su validation; test usato una volta per versione. Test già consultato diventa esplorativo per iterazioni successive.

Fonti: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), [GPT-2](https://huggingface.co/openai-community/gpt2), [BPE](https://huggingface.co/docs/tokenizers/quicktour).
