# Fase3 — dubbi e ricerca

Test fermati su richiesta utente. Soglia5 linguaggio:512 update eseguiti, stop durante validation; confronto CE/generazione incompleto. No riavvio.

Ordine operativo nuovo dei numeri originali: **8 →1 →2 →6 →3 →4 →5 →7**. Suite e test: [SUITE_DIAGNOSTICA_FASE_3.md](SUITE_DIAGNOSTICA_FASE_3.md). Numerazione sotto tenuta per storia.

## 1. Stiamo giudicando il modello con troppo pochi dati?

**Sì, per capacità linguistica.**64 storie/8126 target = diagnostica. TinyStories mostra linguaggio coerente ma con piccoli GPT-Neo, contesto512, training scala diversa; non valida questo connettoma. Pochi update falliti non bocciano architettura/dataset. [TinyStories, §§1/4](https://arxiv.org/html/2305.07759v2).

Da chiarire: curva su più dati nuovi contro baseline semplice a stesso budget; non ripetere stessi esempi.

## 2. Il token resta distinguibile dopo il grafo?

**Non misurato. Priorità alta.**17.937 ingressi → grafo →2241 neuroni letti →77 gruppi →154 caratteristiche. Tanti neuroni non garantiscono trasmissione utile. Letteratura GNN vede strozzature nella propagazione lunga; applicarlo al nostro LIF è ipotesi, non diagnosi. [Alon/Yahav](https://arxiv.org/abs/2006.05205).

Da chiarire: identità token decodificabile dalle rappresentazioni? Prima copia a ritardo0, poi memoria. No bypass auto.

## 3. L'attenzione usa informazione utile?

**Gradienti presenti, contributo non provato.**Mappe/pesi attenzione da soli non provano utilità causale. Letteratura propone controlli uniformi e più semi, interpretazioni discusse. [Wiegreffe/Pinter](https://arxiv.org/abs/1908.04626).

Da chiarire: diversità K/V, sensibilità al contesto, confronto con memoria perturbata; ablazioni allenate separate per attribuire vantaggio.

## 4. TBPTT16 è stato confrontato equamente?

**A dati uguali sì; come ottimizzazione no.**10 epoche:120 update aL16 contro200/200/160 aL8. Stesso LR, no calibrazione dedicata. TBPTT dà bias del gradiente; finestra più lunga non garantisce miglioramento a ogni budget. [Aicher et al.](https://proceedings.mlr.press/v115/aicher20a.html).

Da chiarire: separare lunghezza gradiente, frequenza update e budget. No bocciatura diL16.

## 5. Virgole ripetute: decoding o modello?

**Entrambi possibili; non solo greedy.**Massimizzazione in decoding può dare ripetizioni pure con LM validi. [Holtzman et al.](https://arxiv.org/abs/1904.09751).

Qui run soglia10/512 update predice virgola su2032/2048 target pure col testo reale in ingresso; CE6,035 contro unigramma6,012. Segno locale di previsione poco condizionata, non prova del punto esatto dove si perde info. Sampling diagnostica decoding; non certifica apprendimento né basta a risolvere.

## 6. Dinamica e gradienti LIF sono ben calibrati?

**Finitezza verificata; calibrazione no.**Scala del gradiente surrogato può contare più della forma. [Zenke/Vogels](https://pubmed.ncbi.nlm.nih.gov/33513328/). Inizializzazione rispetto alle fluttuazioni del potenziale può aiutare propagazione/apprendimento pure con segni vincolati. [Rossbroich et al.](https://zenkelab.org/2022/06/fluctuation-driven-initialization-for-spiking-neural-network-training/).

Da chiarire: potenziale rispetto a soglia, distribuzione gradienti per regione, scala feedback/token. Modifiche gain/reset/leak servono proposta esplicita.

## 7. Più archi = più capacità utile?

**Non automaticamente.**Soglia5 tiene2,27× archi, ma il nostro loader rinormalizza i pesi entranti: cambia anche il peso dei percorsi vecchi. Memoria testata non migliora molto; un solo seme e training breve non provano inferiorità generale. Le strozzature dipendono da topologia/aggregazione, non dal solo conteggio archi. [Alon/Yahav](https://arxiv.org/abs/2006.05205).

Da chiarire: percorsi sensoriali→readout e segnale reale, poi confronto di apprendimento più informativo. CE soglia5 finale manca perché fermata dall'utente.

## 8. L'80% deve bloccare tutto?

**Scelta del nostro protocollo, non soglia scientifica dalle fonti.**[Protocollo locale](PROTOCOLLO_FASE_3.md). Restano veri i fallimenti rispetto a quel criterio; non implicano impossibilità di apprendere linguaggio.

Proposta, non applicata: distinguere correttezza/stabilità indispensabili, diagnostica memoria, qualità linguistica da pilot. Calibrare task/budget con controllo semplice prima di renderlo requisito universale. Tenere tutti gli esiti storici.

## Cosa SpikeGPT chiarisce

SpikeGPT usa blocchi fatti per linguaggio: Spiking RWKV, channel mixer, residual e token shift; non un connettoma anatomico fisso. La variante216M usa BPE e toglie il binary embedding esplicito. Il paper riporta training12/48 ore su4 V100 per45M/216M: prova di fattibilità SNN, non ricetta trasferibile né requisito hardware per noi. [SpikeGPT, §§3/4.3](https://arxiv.org/html/2302.13939v4).

## Ordine proposto

1. Chiarire passaggio d'info token→readout con prova semplice e controllo.
2. Se passa, pilot LM con più dati nuovi e confronto unigramma/modello semplice; budget concordato prima.
3. Diagnosticare memoria/attenzione/dinamica dal punto dove manca info.

No altro training, modifica architetturale o abbassamento criterio in questa ricerca. Ricerca orientativa: fonti pertinenti, nessuna garanzia di trasferibilità al cervello della mosca.
