# Profilo paging — opt-in utente

Autorizzazione esplicita sessione: «Prosegui con paging; accetto rallentamenti».

Registro Windows: `C:\pagefile.sys`, config16384–32768MB. GlobalMemoryStatusEx: limite commit ~29,90GiB al controllo, margine ~12,65GiB; valori dinamici, no uso SSD misurato. Config sistema non modificata.

`--allow-paging` abilita guardie alternative solo in nuovi runner fase3:

- Lancio: RAM libera>=0,125GiB emergenza e margine commit>=6GiB; VRAM totale<=6500MiB. Soglia fisica allineata al runtime dopo stop prudenziale a0,547GiB liberi nonostante9,40GiB margine commit.
- Runtime: RAM libera>=0,125GiB emergenza, margine commit>=4GiB, VRAM totale<=7000MiB.
- Timeout totale600s; no avanzamento90s -> abort. Cinque update consecutivi oltre2s -> abort.
- No offload esplicito pesi su CPU; calcolo modello GPU. Windows può paginare memoria host paginabile. Profilo non garantisce assenza rallentamenti, non forza quali pagine vadano su SSD.
- Default originale preservato per comandi senza flag. Risultati riportano profilo e headroom effettivi; non confondere paging autorizzato con miglioramento efficienza modello.

Ottimizzazione runner: valutazione con graph1 token riutilizzato anziché graph8 token, confronto valori/stato con riferimento GPU eager. Prefissi sintetici senza target consumati in inferenza per costruire contesto, senza aggiornare Adam. Batch interamente vuoto ordinario resta caso distinto dai prefissi non scored.

Tentativi iniziali abortiti da vecchie guardie conservati; nessuno score parziale usato per selezione.

Rettifica dopo primo tentativo paging: soglia fisica0,5GiB aveva fermato il run con quasi10GiB margine commit e passi ancora regolari (~0,34s). Ridotta a0,125GiB emergenza nell'opt-in paging; commit/stalli restano controlli principali. Run interrotto non è risultato di apprendimento, non modifica budget o criteri.

Griglia3.4: `phase3_grid_hot.py` riusa runtime in sequenza per ridurre caricamenti/paging. Pesi ripristinati e optimizer ricreato per ogni caso; hash iniziali confrontati per seme, allocatore riportato al baseline residente fra casi e0 finale. Timer600s per caso, supervisore unico con guardie continue; massimo7200s per contenitore24 casi. No caso GPU parallelo.
