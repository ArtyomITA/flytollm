# Report Luna — SNN, linguaggio, grafo mosca

Data: 2026-09-13. Ricerca bibliografica, nessun codice modificato, nessun GPU/test/download pesante.

## Esito operativo

Letteratura mostra **fattibilità**, non ricetta universale: SNN addestrati direttamente fanno riconoscimento vocale e generazione/testo, ma casi riusciti usano architetture per memoria/propagazione e budget molto maggiori del nostro smoke. SpikeGPT: 45M/216M parametri, Adam, lr 6e-4, 12/48 h su 4×V100; BPE solo variante 216M, pretraining OpenWebText2. Variante 45M usa binary embedding; 216M lo rimuove, usa neuroni primo layer per encoding. Risultato: prova che SNN-LM può apprendere, non che grafo mosca debba né che 200 update bastino. [SpikeGPT, metodi/setting](https://arxiv.org/html/2302.13939v4#sec-3-6), [setting/risultati](https://arxiv.org/html/2302.13939v4#sec-4-3)

T1 locale: GRU d0 DEV100% → identità immediata apprendibile; d1 DEV30,8% dopo 200 update → controllo memoria/budget non calibrato, non impossibilità. Bigramma DEV CE4,803 vs unigramma6,012 → contesto locale contiene segnale. Mosca storica CE5,918–6,149 non supera baseline: evidenza locale, non verdetto architetturale.

Aggiornamento: T2/T3 eseguiti dopo; metriche e lettura nel resoconto separato [RESOCONTO_T2_T3.md](RESOCONTO_T2_T3.md), senza riaprire qui la suite.

## Filone 1 — identità, memoria, linguaggio in altri SNN

| Lavoro/proposta | Evidenza pubblicata | Pertinenza T0/T1 | Rischio/limite; diagnostica minima |
|---|---|---|---|
| SpikeGPT: token-mixer Spiking-RWKV + channel-mixer SRFFN, residual loop | RWKV ricorrente mantiene stato A/B con decay; residual; binary embedding con surrogate arctan solo nella 45M; 216M lo rimuove, usa neuroni primo layer; LM autoregressivo; readout proietta con matrice embedding trasposta (fonte non dimostra condivisione parametro). 45M: Enwik8 test BPC 1,262 (T=3072); 216M pretrain: WikiText-2 test PPL18,01, WikiText-103 39,75. | Identità non affidata al solo firing: token shift porta contesto vicino; stato ricorrente conserva storia; readout denso legge rappresentazione. Suggerisce separare d0 da memoria, come suite T2. | Architettura artificiale, non connettoma; 12/48 h, 4 V100, corpus milioni; binarizzazione/embedding diversa. Diagnostica: probe congelato su embedding→correnti→154 feature→z256→logits; d0 con contesti nuovi; CE/acc, non solo firing. |
| SpikeLM: elastic bi-spike | Meccanismo fully spiking con direzione, ampiezza e frequenza elastiche; autori: spike binario singolo non codifica semantica sufficiente; validato su compiti generativi/discriminativi. | Rischio diretto per grafo: spike binario + pooling può perdere identità. Prima misurare informazione; solo dopo ipotizzare multi-bit/rate. | Non prova che multi-bit serva qui; architettura/dati non comparabili. Diagnostica: probe separati su potenziale, rate, spike e concatenazione già usata; decidere da separabilità train/DEV, senza cambiare subito porte. |
| Ponghiran–Roy: LIF con dinamica ricorrente migliorata | Modifica dinamica interna; output multi-bit; affronta vanishing gradient. Su TIMIT/LibriSpeech: accuratezza entro 1,10%/0,36% da LSTM, 2× meno parametri; sparse output riduce moltiplicazioni 10,13×/11,14× vs GRU. | Distingue “stato LIF esiste” da “stato addestrabile”: d1=30,8% richiede calibrazione di leak/soglia/gradiente prima di giudicare grafo. | Speech encoder ≠ LM autoregressivo; multi-bit cambia contratto. Diagnostica T3: potenziale−soglia, firing valido, gradiente per regione e rapporto update/peso; nessun tuning combinatorio. [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20771) |
| Bittar–Garner: LIF standard, surrogate, encoder LVCSR | LIF standard senza gate/multi-bit sostituisce encoder LSTM con perdita piccola; autori: robustezza a gradienti esplosivi vs RNN gated. | Mantiene ipotesi semplice: non serve aggiungere gate per testare memoria; serve però sequenza/dati adeguati. | Encoder vocale con decoder esterno; non dimostra readout LM dal grafo. Diagnostica: controllo d1 con stessi stream/budget; separare gradiente token-sorgente da gradiente embedding condiviso. [arXiv 2212.01187](https://arxiv.org/abs/2212.01187) |
| Surrogate/normalizzazione/inizializzazione | Ledinauskas et al.: gradiente surrogate troppo largo → esplosione, troppo stretto → vanishing; tarano larghezza, batch-norm su correnti, AdamW+one-cycle, 10 time-step. | Rafforza T3: finitezza locale non implica scala utile; misura potenziale rispetto soglia e gradiente, poi eventuale singola calibrazione. | Prova su SNN visivi, non trasferibile numericamente a LIF con segni/topologia fissi. Nessun valore universale di gamma/gain. [paper](https://arxiv.org/abs/2006.04436) |
| Residual SEW-ResNet | SEW residual implementa identity mapping, permette >100 layer addestrati; evidenza su visione, non testo. | Residual/skip può proteggere identità attraverso profondità; nel grafo anatomico aggiungerlo cambia modello e conteggio archi. Prima probe; residual solo come variante esplicita. | Non evidenzia che bypassare connettoma aiuti memoria linguistica. Diagnostica: confronto con stessa topologia + residual artificiale, pari parametri/update, solo se T2 localizza perdita. [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html) |

### Cosa trasferire davvero

1. **Readout conta quanto nucleo.** SpikeGPT usa rappresentazione contestuale e readout esplicito; nostro `154 → 256 → Eᵀ` può cancellare identità anche con 166.700 nodi attivi. T2 deve misurare ogni interfaccia, non inferire dal numero di neuroni.
2. **Memoria deve essere leggibile e allenabile.** Stato LIF, cache K/V e contesto consultabile non sono stessa cosa: T4/T5 devono distinguere effetto causale cache, contenuto K/V, finestra BPTT e numero update.
3. **Residual/embedding non sono “biologia”.** Sono adattatori ingegneristici candidati. Prima versione: nessun bypass; verificare se segnale esiste prima/dopo pooling. Nel nostro contratto, weight tying `z256 @ E.T` resta confronto utile; formula SpikeGPT mostra matrice embedding trasposta, non prova condivisione parametro. Gradiente uscita denso può dominare embedding.
4. **Budget va normalizzato.** Confrontare token validi, update, target visti/nuovi, BPTT e tempo GPU. SpikeGPT non autorizza estrapolare 200 update → capacità linguistica; T1 d1 resta aperto.
5. **Surrogate è ipotesi di ottimizzazione.** Monitorare distribuzioni, non solo NaN/Inf. Se gradiente si spegne dopo una regione, localizzare; non cambiare insieme soglia, gain, leak, T e readout.

## Filone 2 — ottimizzare grafo connettoma mosca

Attenzione comparabilità: metriche FlyWire sotto sono del cervello completo di femmina adulta (snapshot/definizioni Nature), mentre progetto usa male-CNS con 166.700 body ID. Conteggi, soglie e ID **non sono trasferibili**; evidenza vale come principio topologico, non numero locale. Progetto conserva oggi tutti gli ID/nodi (166.700), archi soglia10=2.753.975 e soglia5=6.242.118; soglia cambia anche normalizzazione pesi entranti. Input: nodi `sensory`; output: 2.241 neuroni in 77 gruppi →154 segnali. Gruppi sono convenzioni `superclass × rootSide`/blocchi ≤32, non regioni spaziali. Qualunque riduzione deve preservare ID locali, direzione, segno, archi reciproci e mappa porte.

### MaleCNS: cosa è già studiabile sugli ID locali

Fonte primaria del dataset è Berg et al., che descrive connectome completo maschile (cervello, lobi ottici, VNC), con annotazioni curate di classe/tipo, lato, neuromero, nervo ingresso/uscita e circuiti sensoriali→motori; paper riporta 166.691 neuroni, quindi il numero non va sostituito automaticamente con i 166.700 ID caricati dal nostro repository. Pagina dati ufficiale documenta `male-cns:v1.0`, API NeuPrint, ROI neuropil del cervello e ROI VNC rifinite, oltre a tabelle annotazioni, partner sinaptici con neuropilo primario, pesi e predizioni di neurotrasmettitore. [Berg et al., Male CNS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/), [dataset/API ufficiale](https://male-cns.janelia.org/download/)

Nel checkout sono subito interrogabili, senza cambiare codice, gli ID e i campi `superclass`, `class`, `subclass`, `type`, `somaNeuromere`, `entryNerve`, `exitNerve`, `rootSide`, `mancType`, più pesi del connectome e segni da neurotrasmettitore. Consente di enumerare ingressi `sensory`, seguire reachability e percorsi diretti verso `descending_neuron` e classi VNC/motor/efferent, e confrontare t10/t5 mantenendo stessi body ID, segni e porte. `fly_graph.py` usa però solo annotazioni superclass, neurotrasmettitori e pesi: mapping neuropilo esplicito richiede tabella/API syn-partners/ROI ufficiale, non presente come colonna nelle annotazioni locali ispezionate.

Non esiste, nelle fonti MaleCNS consultate, prova pubblicata di training o successo su language modeling: dataset e paper supportano anatomia e circuiti sensorimotori, non identità linguistica o LM autoregressivo. Restano fuori dinamiche recettoriali, neuromodulazione, gap junction e stato interno; ogni analogia con LM è ipotesi locale.

| Proposta | Evidenza primaria | Fedeltà/costo | Rischio; osservazione discriminante |
|---|---|---|---|
| A. Tutti i nodi, selezione archi per peso soglia progressiva | FlyWire: 93,3% neuroni in giant SCC, 98,8% in giant WCC; rimozione per basso grado conserva componente gigante fin quasi alla fine; soglia sinaptica non cambia sostanzialmente metriche topologiche. | Massima fedeltà nodi/ID; t10 già baseline, t5 ~2,27× archi e costo locale storico ~2×. | Soglia non equivale a “importanza funzionale”; rinormalizzazione confonde qualità. Misurare raggiungibilità input→readout, shortest directed paths, SCC, firing/gradiente; stesso seed/budget. [Network statistics](https://www.nature.com/articles/s41586-024-07968-y) |
| B. Conservare rich-club + percorsi sensoriali→output, non top-K grado cieco | Rich-club 40.218 neuroni (~30%), ma rete ha percorsi alternativi; 676 broadcaster/638 integrator; rich-club vicino a molte modalità sensoriali; flow model classifica distanza da olfatto/gusto/visivo. | Riduzione mirata possibile; preserva snodi multisensoriali e propagazione. Costo potenzialmente molto minore, da misurare sui body ID reali. | Rimuovere nodi può rompere feedback/inibizione; degree ranking non cattura segno, dinamica o porte. Osservare quota archi/neuroni conservati, copertura porte, path coverage per ogni input/output, CE/acc d0/d1. [Network statistics, flow/rich-club](https://www.nature.com/articles/s41586-024-07968-y) |
| C. Selezione per neuropili, mantenendo connettori tra regioni | FlyWire analizza 78 neuropili: EB/FB più interni; MB prevalentemente esterno; LH invia più di quanto riceve, lobula plate riceve più di quanto invia; 52% pesi classificabili interni. | Modello a blocchi può ridurre temporanei/porte mantenendo regioni e interfacce. Buona interpretabilità anatomica. | Tagliare neuropili “non linguistici” è ipotesi: MB/central complex hanno circuiti di apprendimento, navigazione, sonno; collegamenti esterni possono essere proprio ponte. Diagnostica: ablation temporanea per neuropilo + variazione logits/CE e reachability, poi eventuale retraining separato. [hemibrain](https://elifesciences.org/articles/57443), [network stats](https://www.nature.com/articles/s41586-024-07968-y) |
| D. Selezione input/output per ruolo, mantenere vie complete | FlyWire consente tracing completo sensoriale→motor output; Shiu et al. costruiscono LIF brainwide con connettività+neurotrasmettitori e predicono trasformazioni sensorimotorie; 91% di 164 predizioni testate coerenti, ma fallimenti in circuiti basali inibitori/neuromodulatori. | Porte anatomiche più mirate, pooling meno arbitrario; preservare grafo intermedio evita shortcut. | Sensorimotor ≠ linguaggio; connettività da sola manca gap junction, non-spiking, stato interno, neuropeptidi, recettori. Misurare non solo activation: informazione token e gradiente fino a output. [Shiu et al. Nature](https://doi.org/10.1038/s41586-024-07763-9) |
| E. Pruning per archi deboli/low-flow, con fallback archi ponte | In FlyWire, 3% coppie vicine fa 71% connessioni; rete mostra small-worldness 141, reciprocity 0,138 e percorsi alternativi. Flow probabilistico usa frazione sinapsi, ma ignora segno. | Può abbassare costo mantenendo archi forti e percorsi corti. | Peso anatomico ≠ efficacia dopo `softplus`, segni e stato LIF; archi deboli aggregati possono portare segnale. Test: confronto topologia e dinamica su stesso checkpoint, misurare differenza logits, firing, gradiente e SCC prima di adozione. [Network statistics](https://www.nature.com/articles/s41586-024-07968-y) |
| F. Coarsening per tipo/neuropilo con espansione readout | Hemibrain mostra che cell-type census e connessioni sono utili, ma FlyWire trova ~1/3 tipi hemibrain non reidentificabili affidabilmente tra connectomi; Shiu mostra limiti di modello uniforme. | Riduce N e temporanei; conserva identità di tipo se mappa multi-connectome validata. | Pooling può mescolare lati, neurotrasmettitori e funzioni; perde identità token. Test: separabilità pre/post pooling e prova etichette permutate; mai chiamare “regione” blocchi artificiali ≤32. [Schlegel et al.](https://www.nature.com/articles/s41586-024-07686-5), [Scheffer et al.](https://elifesciences.org/articles/57443) |

## Osservazioni discriminanti minime

| Osservazione | Lettura prudente | Azione |
|---|---|---|
| `154 → z256` collassa identità già separata a monte | strozzatura pooling/readout | mantenere grafo; confrontare probe per segnale, senza pruning |
| `z256` decodifica ma CE head resta alta | informazione presente, ottimizzazione/head insufficiente | calibrare d1/surrogate/LR prima di cambiare topologia |
| candidato t5/rich-club/flow conserva reachability ma cambia SCC, segni, logits o gradiente | riduzione infedele o dinamicamente diversa | nessuna adozione; ripristinare nodi/archi ponte e validare su male-CNS |

## Cosa resta ignoto

- Nessuna prova che grafo intero batta grafo ridotto su LM; nessuna prova che MB, CX, GNG o rich-club siano “linguistici”.
- Nessun budget calibrato per d1/d4/d8 sul controllo; un seme locale; T6 non eseguito. T2/T3 eseguiti dopo: dettagli nel resoconto separato.
- Non sappiamo se perdita principale sia embedding/porte, pooling, LIF, surrogate, attenzione/cache, TBPTT o head weight-tied.
- Connettività/NT non descrivono recettori, neuromodulazione, gap junction, non-spiking, morfologia o stato basale; modello Shiu esplicita questi limiti.
- “Preservare più mosca” può significare nodi/ID, archi, segni, percorsi, tipi o dinamica: obiettivi diversi, nessuna metrica unica già autorizzata.

## Tre proposte ad alta priorità

**1 — Grafo intero come riferimento:** mantenere tutti i nodi male-CNS e ID; misurare reachability/path coverage input→output, SCC, reciprocity, segni e copertura 256 componenti per t10/t5. Selezione nodi rich-club/flow/neuropili resta **ipotesi diagnostica**, non adozione.

**2 — Interfacce prima del pruning:** su stesso grafo/stream, registrare `E → corrente → letture grezze →154 → z256 → logits` e probe train/DEV sui segnali disponibili. Segnale perso in interfaccia → porte/pooling; segnale leggibile ma CE alta → head/ottimizzazione.

**3 — LM calibrato:** controllo GRU d1 con budget dichiarato; poi T5 L8/L16 a update uguali. T6 solo se percorso informativo passa; confronto unigramma/bigramma su dati nuovi. Nessuna conclusione da 200 update.

## Fonti primarie

- [Zhu et al., SpikeGPT, arXiv:2302.13939](https://arxiv.org/html/2302.13939v4)
- [Xing et al., SpikeLM, ICML/PMLR 235 (2024)](https://proceedings.mlr.press/v235/xing24d.html)
- [Ponghiran & Roy, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20771)
- [Bittar & Garner, surrogate-gradient SNN LVCSR](https://arxiv.org/abs/2212.01187)
- [Ledinauskas et al., Training Deep SNNs](https://arxiv.org/abs/2006.04436)
- [Fang et al., SEW-ResNet, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html)
- [Scheffer et al., hemibrain, eLife 2020](https://elifesciences.org/articles/57443)
- [Dorkenwald et al., FlyWire wiring diagram, Nature 2024](https://doi.org/10.1038/s41586-024-07558-y)
- [Schlegel et al., whole-brain annotation, Nature 2024](https://www.nature.com/articles/s41586-024-07686-5)
- [Lin et al., network statistics, Nature 2024](https://www.nature.com/articles/s41586-024-07968-y)
- [Shiu et al., computational brain model, Nature 2024](https://doi.org/10.1038/s41586-024-07763-9)
- [Berg et al., complete male CNS connectome, bioRxiv/Cell](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)
- [Janelia MaleCNS v1.0, download/API/annotations](https://male-cns.janelia.org/download/)
- [Takemura et al., visual motion circuit, Nature 2013](https://doi.org/10.1038/nature12450)
- [Eichler et al., larval mushroom body connectome, Nature 2017](https://doi.org/10.1038/nature23455)

Classificazione usata: “evidenza pubblicata” = risultato riportato da studio primario; “analogia” = componente architetturale trasferita con cautela; “ipotesi locale” = proposta sul codice/protocollo mosca, da testare.