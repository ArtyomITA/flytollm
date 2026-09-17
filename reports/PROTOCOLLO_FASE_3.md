# Fase3 — protocollo diagnostico v1

Fissato prima run3.1/3.2. Architettura2.5 invariata, CNS soglia10, FP32 CUDA Graph; CPU solo dati/I/O/supervisione. Nessuna variante architetturale autorizzata qui.

## 3.1 / 3.2

- Train: storie0/1 di `dataset/prepared_v1/train`, primi16 target ciascuna (input0..15, target1..16), BOS incluso come input. Campione32 target, non due storie intere. B2,L8, due finestre/epoca; reset stato/cache ogni epoca; carry detached dopo update fra finestre.
- Validation riservata: prime16 voci validation manifest `evaluation_indices.json`; nessuna valutazione/selezione su test finale. In3.2 non si misura generalizzazione.
- Seed89; init interne embedding17/attenzione23 restano quelle architettura. Salvare hash tokenizer/dati/config/sorgenti e ID effettivi campione.
- Adam, betas(0.9,0.999), eps1e-8, decay0, foreach=False, capturable=True. Gruppi nucleo/interfacce/attenzione; stesso LR iniziale. Clipping norma globale1, gradienti finiti controllati. Gate appreso normalmente; nessuna modifica manuale valore iniziale.
- Smoke3.1: due epoche; verifica skip batch senza target validi dopo init momenti, reset/epoca, carry, metriche, salvataggio/caricamento pesi/config.
- Overfit3.2: max200 epoche/400 update, valutazione teacher-forced senza update a epoca0 e ogni20 epoche, stessi32 target e stato iniziale azzerato. Successo: CE finale <=50% CE iniziale, accuracy >=50%, valori finiti, aggiornamenti in tutti i gruppi. Stop anticipato solo a checkpoint che soddisfa entrambi criteri.
- Primo tentativo LR1e-4. Solo se insufficiente, secondo LR1e-3 da identici pesi iniziali, dati/ordine/budget; nessuna estensione nascosta. Fallimento di entrambi resta risultato diagnostico.
- Guardie: <=600s/caso, timeout fase300s, almeno1.75GiB RAM libera al lancio, soglie runtime supervisore esistente, GPU totale<7000MiB; abort NaN/Inf. Un worker GPU alla volta.
- Log: CE/accuracy comparabili a pesi fissi, loss finestre training separata; firing/rate, silenti e firing ogni sottopasso, potenziale, gate, RMS token/feedback sulle porte, gradienti per parametro, delta pesi per gruppo, velocità/memoria. Generazione greedy prima/dopo: prompt primi4 token,12 nuovi, entrambe storie; nessuna selezione estetica.
- Checkpoint diagnostico pesi/config+metadati e hash; non contiene optimizer/RNG/stato per ripresa esatta. Niente pretraining o generalizzazione dichiarati.

## Criteri successivi, fissati prima dei relativi risultati

- 3.3 copia/ritardo: vocabolario sintetico8 simboli ordinari, ritardi1/4/8; stream train/dev/test con semi171/172/173 disgiunti, punteggio solo su posizioni con antecedente disponibile. Baseline chance12.5%; obiettivo accuracy dev>=80% per distanza, max10min/caso. Specifica operativa definitiva del task prima di avviarlo; nessun test sintetico per tuning.
- 3.4: Adam/AdamW/Muon/ROOT, tre LR per famiglia predefiniti prima del confronto e due semi17/23, <=10min/caso. Confronti a update uguali e tempo uguale distinti; scelta validation, margine CE>=2% consistente nei due semi per sostenere un vantaggio. Altrimenti inconcludente, baseline mantenuta. ROOT coefficienti fallback espliciti.
- 3.5: run entro10min, evaluation inizio/fine sulle16 storie validation riservate, prefissi128 target senza superare EOS; obiettivo CE validation ridotta>=5% vs init, nessun NaN/Inf o deriva persistente metriche. Non implica comprensione generale. Se mancato, fase3 resta aperta.

Modificare un criterio richiede nuova versione motivata prima dei nuovi risultati, conservando esiti precedenti.
