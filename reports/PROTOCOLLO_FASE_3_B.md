# Fase3.3–3.5 — dettagli operativi v1

Addendum protocollo3; fissato prima nuovi risultati. Architettura invariata, GPU FP32/CUDA Graph, B2/L8, clipping1, guardie esistenti; <=600s/caso. No tuning su test finale. Esiti negativi no autorizzano varianti architetturali, no rinominati successi.

## 3.3

- Stress256 token: checkpoint3.2, payload train0 ripetuto senza BOS/EOS interni, BOS iniziale; no reset fino256. No testo naturale continuo. Verificare cache128, finitezza, potenziali nel tempo; no update.
- Copia ritardata: ID400..407, 8 simboli uniformi indipendenti. Input BOS +32 simboli, target posizione t = simbolo t-d, d=1/4/8; primi d simboli e BOS no scored. PAD fino40 posizioni per forme statiche. Reset per stream, carry detached ogni8.
- Train8 stream (seed171), dev8 stream (seed172); rigenerati deterministicamente per distanza, assert no stream identici fra split. Seed173 riservato test sintetico, no usato. 10 epoche,200 update massimi; Adam LR1e-3. Init89 con mappature anatomiche fisse.
- Loss su vocabolario completo4096; accuracy primaria argmax ristretto agli8 simboli (chance12,5%), accuracy completa riportata separatamente. Obiettivo dev>=80% per distanza; confrontare prima/dopo. Fallimento = memoria no dimostrata entro budget, no impossibilità teorica.

## 3.4

- Famiglie Adam, AdamW, Muon+AdamW, ROOT-fallback+AdamW. LR base1e-4/3e-4/1e-3; Muon/ROOT scala matrici0,2√max(shape). Betas Adam/AdamW(0,9;0,999), decay0 per tutti. Adam/AdamW equivalenti nel principio: controllo, no confronto weight decay.
- Due semi17/23: init pesi casuali embedding, porte e proiezioni; anche ordine dati. Topologia, segni, norme, bias/gate iniziali e mappature anatomiche invariati. Ogni famiglia/LR parte da stessi pesi per seme.
- Train16 storie, indici2..17, primi32 target: quattro finestre per coppia,32 update/epoca. Tre epoche =96 update,1536 target massimi. Escludere padding dal conteggio.
- Validation selezione: prime4 storie dei16 indici riservati, primi32 target; no update. CE aggregata per target, accuracy, risorse; iniziale/finale. No usare questi4 esempi come test imparziale del modello scelto.
- Confronto dati uguali a96 update. Snapshot tempo uguale a ultimo update terminato entro20 secondi cumulativi training (staging/sync inclusi, caricamento/evaluation esclusi); validation snapshot eseguita fuori cronometro. Due confronti separati.
- Tre LR per famiglia scelti tramite CE validation media dei2 semi. Vantaggio sostenuto solo se riduzione>=2% vs miglior Adam in entrambi semi; altrimenti inconcludente. No sostituzione automatica baseline.

## 3.5

- Baseline Adam resta riferimento salvo decisione esplicita successiva. Run da init17, LR1e-4; no init da pesi overfit3.2 né scegliere ex post seme migliore.
- Train indici18..81 (64 storie), prefissi128 target, B2/L8, reset per storia/carry detached fra finestre. Un passaggio =512 update massimi; stop training a240s se precedente. Totale inclusa validation entro600s.
- Validation prima/dopo: tutti16 indici riservati, prefissi128, senza superare EOS. Evidenziare separatamente ultimi12 no usati per selezione3.4; test finale resta escluso.
- Successo diagnostico: CE validation ridotta>=5%, no NaN/Inf né deriva persistente. Registrare potenziali, firing, gate/correnti e norma gradienti ogni32 update. Segnalare deriva se RMS potenziale cresce consecutivamente nelle ultime3 misure oltre2× mediana prime3; confronti fra storie diverse sono allarmi diagnostici, no prova instabilità.
- Se memoria3.3 o stabilità3.5 no superano criteri, fase3 resta aperta anche avendo eseguito tutti run richiesti. No avviare pretraining né cambiare architettura automaticamente.
