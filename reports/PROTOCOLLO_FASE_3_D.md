# Confronto TBPTT e soglia10/5

Utente autorizza: TBPTT16 se serve; poi soglia10 vs5 a dati/budget uguali. No adozione auto. Protocolli/risultati precedenti tenuti.

## TBPTT

- Memoria: stessi8 stream train171/dev172, seed89, Adam0,001, distanze1/4/8.10 epoche, stesso num target; confronto TBPTT8 (checkpoint epoca10 provaC) vs16 fresh. Update differiscono perché finestra cambia; esplicitare.
- TBPTT16:48 posizioni padded invece40;8 PAD extra, stato congelato, no loss extra. Stessi32 simboli reali/BOS. T biologico8=4pre+4post invariato.
- Verifica cattura16: loss/gradienti/update vs riferimento eager GPU, ripristino pesi/momenti/stato pre-training.

## Soglia

- Memoria: soglia10 vs5, TBPTT16,10 epoche, stesso LR/seme/stream/num target. Accuracy train/dev completa e ristretta, CE, tempo training, picco allocator GPU.
- Linguaggio: soglia10 vs5, TBPTT8, seed17, Adam0,0003, train64 storie18..81 prefissi128,1 passaggio=512 update max; target effettivi uguali. Validation16 riservate + ultime12 separate; stessi2 prompt greedy24.
- Test finale chiuso. Parametri artificiali iniziali/mappature identici; stessi166.700 nodi e segni. Più archi = più parametri core e rinormalizzazione pesi entranti: confronto config complete, non effetto isolato del solo num archi.
- Confronto primario a dati/passaggi uguali. Limite600s/caso e400s training; se guardia/timecap interrompe prima, confronto dichiarato incompleto, no conclusioni a budget uguale.
- Soglia5 resta sperimentale anche con numeri migliori. Criterio memoria80% invariato; CE−5% e stabilità restano diagnostici, non prova linguaggio coerente.
- Un processo GPU alla volta, watchdog/paging autorizzato; no offload modello CPU. Tempo training include staging/sync; non CUDA-event puro. Riportare correttamente.
