# Fase3 — recupero v2

Utente vuole: riprovare chiusura fase3 dopo taglio carico browser/gioco. Risultati v1 tenuti. Zero modifica architettura; GPU FP32/CUDA Graph; B2/L8, Adam, clip1, modello uguale. Un worker per volta,600s/caso, guardie paging già ok.

## Memoria3.3

- Stessi stream train171/dev172, seed pesi89, LR0,001, distanze1/4/8; test173 chiuso.
- Budget30 epoche non10:600 update max (480 per ritardo8, prefissi senza target esclusi).
- Valuta train/dev inizio e ogni10 epoche; stop solo se accuracy dev ristretta≥80%. Riporta accuracy vocabolario pieno. Criterio non cala.
- Train alto/dev basso → overfit; entrambi bassi → apprendimento scarso nel budget, isola causa.

## Linguaggio3.5

- Fresh seed17; Adam LR0,0003 (miglior Adam griglia3.4);2 passaggi su stesse64 storie train18..81, prefissi128. Max1024 update, stop training400s, caso600s.
- Validation16 storie a parte, ultimi12 separati; test finale chiuso. Prima/dopo: CE, accuracy, distribuzione argmax, frequenza punto, entropia, confronto unigramma add-one fatto SOLO su target train.
- Generazione: stessi2 prompt validation14447/9560,8 token prompt,24 token greedy. Zero cambio decoding per nascondere collasso.
- Criterio numerico originale: CE−5% e valori finiti, allarme drift originale. Generazione senza ripetizione costante serve per togliere problema qualitativo specifico, ma non basta per dire linguaggio coerente.
- Numeri registrati anche se peggio. Fallimento → fase3 aperta, diagnosi/proposta esplicita; zero tuning illimitato, pretraining o cambio architettura auto.
