# T2/T3 — identità presente; lettura da calibrare

T2/T3 eseguiti. **Token attraversa grafo:** probe separato legge identità da readout grezzo92,29% DEV, da rappresentazione finale84,77%; head originale18,85%. Priorità: lettura/ottimizzazione, prima di tagliare nodi. Non prova head strutturalmente sbagliata: probe ha task8 classi, standardizzazione e training diversi.

Mosca invariata. Nessun nodo/arco rimosso; nessun bypass aggiunto. Audit64 e test linguistico non aperti.

## T2 — training d0

Soglia10, T4+4, B2/L8, Adam0,0001, seed89.80 stream train bilanciati×32 simboli, un passaggio; DEV32 stream distinti.200 update,2.560 target,64,71s training; picco allocator394,83MiB. Checkpoint iniziale/finale salvati. Non confronto equo con GRU T1: dati bilanciati, numero target, LR e BPTT diversi.

| Misura head originale4096 | Prima DEV | Dopo train32 | Dopo DEV32 |
|---|---:|---:|---:|
|CE|8,4184|2,1527|2,1534|
|Accuracy4096|0%|20,51%|18,85%|
|Accuracy ristretta8|13,77%|20,51%|18,85%|

Chance8=12,5%; CE uniforme sulle8 classi=ln8≈2,079. CE finale2,153: gran parte del calo iniziale può venire dall'imparare quali8 token esistono; non scambiare per identità risolta.

## T2 — dove passa informazione

Checkpoint finale congelato. Raccolta1.024 posizioni train +1.024DEV in contesti differenti; simboli bilanciati. Probe affine8 classi, Adam0,03/100 update, standardizzazione solo train. **Ogni probe vede102.400 target ripetuti**, contro2.560 target nuovi nel training mosca: misura accessibilità segnale, non confronto equo di apprendimento. Tutto GPU/CUDA Graph; probe esterno, non inserito nel modello.

| Punto letto | Dimensione | Probe train | Probe DEV | DEV con etichette train permutate |
|---|---:|---:|---:|---:|
|Embedding normalizzato|256|100%|100%|0%|
|Corrente sensoriale|17.937|100%|100%|25%|
|Uscite grezze pre: voltage+rate|4.482|100%|81,15%|9,77%|
|Pooling pre|154|94,04%|60,74%|12,21%|
|Rappresentazione pre|256|87,79%|62,40%|11,04%|
|Uscite grezze post: voltage+rate|4.482|100%|92,29%|11,52%|
|Pooling post|154|100%|88,28%|12,60%|
|Rappresentazione finale|256|100%|84,77%|11,62%|
|Vettore logits, riletto da probe|4.096|100%|80,47%|13,67%|

Controllo costante:12,5% train/DEV. Embedding e corrente hanno solo8 rappresentazioni uniche: con un'unica permutazione, controlli possono dare0%/25%; non sono1.024 predizioni indipendenti. Nei segnali contestuali i controlli permutati restano9,77–13,67%. Un seme; nessun intervallo di confidenza.

Reset fra token isolati: tutte9 tappe conservano8 vettori distinti. Distanza minima grezza post2,441; pooled post0,0925; rappresentazione finale2,888. Distanze fra spazi diversi **non confrontabili direttamente**. In contesto variabile, rapporto energia fra centroidi/entro token post: grezzo0,0624; pooled0,0776; finale0,1038. Metrica aggregata, non misura diretta di informazione.

**Lettura:** perdita di decodificabilità soprattutto pooling pre; fase post recupera. Pooling finale conserva segnale forte. Perfino logits contengono informazione leggibile da altro classificatore, mentre argmax originale fallisce spesso. Non separabili ancora budget, normalizzazione, head condivisa e ottimizzazione; nessuna prova che serva rifare il grafo.

## T3 — dinamica e gradienti

Tre gruppi funzionali disgiunti, **non neuropili**. Medie sulle2.048 posizioni simbolo, PAD/BOS esclusi. Firing = frazione media di sottopassi con spike, non percentuale di neuroni permanentemente attivi.

| Gruppo | Nodi | Firing pre | Firing post | Voltage RMS post | Rate zero nella fase post |
|---|---:|---:|---:|---:|---:|
|Ingressi sensoriali|17.937|43,60%|44,98%|0,597|19,43%|
|Uscite selezionate|2.241|0,356%|0,366%|0,260|98,54%|
|Resto|146.522|0,201%|0,211%|0,413|99,16%|

Uscite poco spiking ma voltage informativo: probe92,29% sulle letture grezze. Firing basso ≠ informazione assente. Potenziale pre-reset misurato negli8 sottopassi; ingressi vicino soglia[0,9;1,1]≈15,4–20,1%, uscite≈1,46–1,58%. Nessuna saturazione globale uniforme; distribuzione disomogenea.

Corrente token RMS0,701; feedback già moltiplicato gate0,249: rapporto delle medie≈35,6%. Feedback significativo, non dominante in RMS medio. Non dimostra utilità attenzione né esclude dominanza locale: serve T4 per interventi sulla cache.

Gradiente CE ultimo token su finestra **BOS+7 simboli**, due stream DEV; stesso checkpoint, nessun update:

- Corrente sensoriale token corrente RMS1,55e−4; token precedente7,78e−5;6 token prima1,60e−5; BOS7 passi prima1,05e−5.
- Gradiente sensoriale nonzero≈89,80% componenti; attenuazione verso passato, non annullamento in questa finestra. Non prova memoria d8 o gradienti oltre TBPTT.
- Gradienti Q/K/V presenti e finiti. Presenza gradiente ≠ memoria utile.
- JSON riporta anche derivata rispetto a componenti corrente fuori porte sensoriali: sensibilità a **iniezioni ipotetiche**, non realizzabili dall'iniettore; non chiamarle gradienti di input allenabili.
- Cambiamento relativo L2 pesi sinaptici effettivi0,113%; parametri raw0,0327%; interfacce7,85%; attenzione9,65%. Parametrizzazioni/scale diverse: rapporti non misurano utilità o quantità d'informazione appresa.

Generazione diagnostica: prompt8 simboli,16 greedy; tende a407. Modello allenato d0, quindi non test linguistico. Voltage RMS post durante16 passi: sensoriali0,622–0,806; uscite0,265–0,274; resto0,427–0,453. Potenziale pre-reset minimo−13,85, massimo2,96: finito, ma coda negativa da considerare prima di run lunghi. Non dimostra stabilità di pretraining.

## Verifiche e risorse

- Training, raccolta/probe, verifica:3 worker separati, tutti `ok=true`; cleanup0MiB, nessun guard abort/offload/fallback.
- Raccolta/probe/gradienti≈32,02s, picco allocator692,94MiB. Memoria include strumentazione/probe; non baseline throughput LM.
- Strumentazione contro forward originale su8 prefissi B2 con PAD e reset misti: logits max errore1,43e−6, tutti tensori stato entro tolleranza2e−4/2e−5. Telemetria finita.
- Head raccolta e valutatore originale coincidono: train20,5078%, DEV18,8477% sugli stessi1.024 target/split.
- Hash sorgenti, protocollo, stream, checkpoint verificati. Nessun training ripetuto; controllo aggiuntivo solo lettura/equivalenza.
- Checkpoint diagnostici contengono pesi/config, non optimizer/RNG per resume esatto. Feature intermedie non archiviate; riproducibili da checkpoint+manifest.

## Decisione e ricerca Luna

T2/T3 chiusi come diagnosi, **fase3 ancora aperta**. Prossimo candidato: calibrare apprendimento della lettura originale, mantenendo grafo e porte; proposta di protocollo prima di eseguirla. T4/T5 per memoria restano da fare; nessun T6/pretraining avviato.

[Report Luna high](REPORT_LUNA_SPIKE_GRAFO.md): SpikeGPT/SpikeLM, dinamiche surrogate, selezione regioni/percorsi. Fonti FlyWire femmina non trasferibili direttamente al male-CNS. Nostri risultati spostano priorità verso lettura/ottimizzazione; selezione nodi resta ipotesi successiva, non modifica approvata.

Evidenze: [training](results/phase3_t2_train.json), [diagnosi](results/phase3_t23_diagnose.json), [equivalenza](results/phase3_t23_verify.json), [protocollo](PROTOCOLLO_T2_T3.md), [stream](configs/phase3_t23_manifest.json).