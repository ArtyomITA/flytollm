# Recupero fase3 — TBPTT e soglie

Stato: TEST FERMATI, utente chiese. Memoria fatta; soglia5 linguaggio rotto durante validation dopo512 update, confronto CE/generazione mezzo. Fase3 aperta. Ricerca: [DUBBI_FASE_3_RICERCA.md](DUBBI_FASE_3_RICERCA.md).

## Più training, TBPTT8

| Ritardo | Train30 epoche | Dev10 epoche | Dev30 epoche |
|---|---:|---:|---:|
| 1 | 83,06% | 31,05% | 72,58% |
| 4 | 26,34% | 12,05% | 14,73% |
| 8 | 19,79% | 9,375% | 9,375% |

Criterio80% dev: nessuno superato. Ritardo1 migliora;4/8 insufficienti anche su train. Non solo overfit. Stream/pesi/LR uguali;600 update per1/4,480 per8.

Linguaggio: fresh seed17, Adam0,0003,1024 update. CE6,1494, peggio di baseline unigramma add-one6,0122 fatto solo su train. Greedy:24 virgole su due prompt fissi. Argmax teacher-forced:1970 virgole/2048 target. Punti =7,38% target train; distribuzione appresa non fa linguaggio buono.

RMS: picco2,18, fine0,817; start~0,42. Crescita transitoria chiara, ma criterio fissato di crescita monotona ultime3 letture non scatta. Pass numerico vs init ≠ stabilità lunga/qualità basta.

## TBPTT8 vs16, soglia10

| Ritardo | TBPTT8,10 epoche | TBPTT16,10 epoche |
|---|---:|---:|
| 1 | 31,05% | 24,19% |
| 4 | 12,05% | 10,71% |
| 8 | 9,375% | 9,375% |

Stessi stream/target/passaggi, LR e init. Finestra16 fa120 update contro200/200/160 a finestra8: confronto a dati uguali, non update uguali. Nessuna ricerca LR per16. Male nel budget, non impossibile in teoria.

Con TBPTT8, simbolo sorgente e target cadono in stessa finestra per27/31,15/28,0/24 posizioni ai ritardi1/4/8. TBPTT16 rimette parte dei collegamenti gradiente, ma non basta qui. Cache128 e gradiente128 restano cose distinte.

## Soglia10 vs5 — memoria, TBPTT16

| Ritardo | Accuracy10 | Accuracy5 | CE10 | CE5 | Training10 | Training5 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 24,19% | 18,55% | 1,9390 | 1,9729 | 77,7s | 152,8s |
| 4 | 10,71% | 11,16% | 2,1153 | 2,1201 | 77,5s | 156,5s |
| 8 | 9,375% | 9,375% | 2,14675 | 2,14659 | 78,0s | 156,7s |

120 update/caso, stessi target/passaggi, stessi pesi artificiali iniziali, verificati via hash. Ritardo4: un solo target giusto in più su224; nessun vantaggio vero. Un seme/campione piccolo, niente conclusione universale.

Soglia10:2.753.975 archi. Soglia5:6.242.118,2,27×. Uguali166.700 neuroni, ID, segni e porte sensoriali verificati. Più archi = rinormalizza pesi entranti: non isola effetto delle sole connessioni aggiunte.

## Correttezza e limiti

- Forward/loss/gradienti/update catturati confrontati con riferimento GPU eager; pesi/momenti/stato rimessi prima dei run. Modello in GPU FP32/CUDA Graph, nessun offload.
- Warning PyTorch durante riferimento eager dopo cattura: stream AccumulateGrad diverso. Equivalenza verificata; nessun fallback. Tempi training misurati sui replay.
- Tempo training include staging/sincronizzazione, non CUDA-event puro. Picchi allocator includono verifiche, training e valutazioni; distinti da memoria totale driver/browser.
- Primo avvioC fermato per fixare riferimento residuo in pulizia, prima di usare risultati. Soglia5/ritardo4 primo tentativo fermato da timeout90s fra log: nessun risultato usato. Runner v2 batte heartbeat ogni32 update; metriche restano ogni64, equazioni/budget uguali.
- Tutti run validi finora:cleanup allocator0. Tentativi abortiti tenuti. Test finale chiuso.

Protocolli: [C](PROTOCOLLO_FASE_3_C.md), [D](PROTOCOLLO_FASE_3_D.md). Evidenze: [comparazione](results/phase3d_summary.json), [CSV](results/phase3d_comparison.csv), [anatomia](results/phase3d_graph_preflight.json), [linguaggio1024 update](results/phase3c_language.json).