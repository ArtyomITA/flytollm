"""Local prose-only compression; preserve audited tokens and original backup."""
from pathlib import Path
import re, json

p = Path('PIANO_FASI.md')
original = p.read_text(encoding='utf8')
replacements = {
    'Ogni cambiamento architetturale proposto va discusso con domanda all’utente prima dell’applicazione. Valutazioni inserite nel piano non autorizzano automaticamente varianti.': 'Modifiche architetturali: domanda all’utente prima di applicare. Valutazioni nel piano ≠ autorizzazione varianti.',
    'Fissare EOS/BOS/PAD, round-trip, tokenizer immutabile per run; pretokenizzare su disco senza caricare intero corpus in RAM.': 'Fissare EOS/BOS/PAD, round-trip, tokenizer immutabile/run; pretokenizzare su disco, corpus fuori RAM.',
    'Scegliere benchmark, piccoli modelli di confronto, split valutazione, prompt/decoding e budget prima dei risultati.': 'Prima dei risultati: fissare benchmark, piccoli modelli confronto, split valutazione, prompt/decoding, budget.',
    'Mantenere operazioni catturabili CUDA Graph, forme/buffer statici e maschere dinamiche nei tensori.': 'CUDA Graph: operazioni catturabili, forme/buffer statici, maschere dinamiche nei tensori.',
    'Misurare firing/saturazione, scale token/feedback, gate, gradienti Q/K/V e raggiungibilità del readout; memoria128 versus finestra TBPTT esplicita.': 'Misurare firing/saturazione, scale token/feedback, gate, gradienti Q/K/V, raggiungibilità readout; distinguere memoria128/finestra TBPTT.',
    'Ogni modifica numerica/architetturale fase4 → ripetere verifiche pertinenti fase3.': 'Modifica numerica/architetturale fase4 → ripetere verifiche pertinenti fase3.',
    'Fissare campione train, subset validation separato, semi, budget e criteri numerici di successo/stop prima dei run. Test finale escluso.': 'Prima dei run: campione train, validation separata, semi, budget, criteri numerici successo/stop. Test finale escluso.',
    'Allenare ripetutamente un piccolo campione train con baseline Adam; confrontare loss iniziale/finale e completamenti sugli stessi prompt.': 'Ripetere piccolo campione train, baseline Adam; confrontare loss iniziale/finale, completamenti stessi prompt.',
    'Controllare che nucleo, interfacce e attenzione ricevano aggiornamenti utili; distinguere loss in calo da semplice attività degli spike.': 'Verificare aggiornamenti utili nucleo/interfacce/attenzione; loss in calo ≠ attività spike.',
    'Controllare gradiente e aggiornamenti Q/K/V/O e feedback; la loro presenza non dimostra da sola un vantaggio dell\'attenzione.': 'Verificare gradiente/aggiornamenti Q/K/V/O e feedback; presenza ≠ vantaggio attenzione.',
    'Nessuna sostituzione automatica della baseline; riportare risultato e motivazione della scelta proposta.': 'Baseline: nessuna sostituzione automatica. Riportare risultato + motivo proposta.',
    'Non scegliere sulla sola ultima loss train.': 'Ultima loss train insufficiente per scegliere.',
    'Congelare configurazione risultante, checkpoint diagnostico e log; indicare priorità per efficienza/recupero archi fase4. Checkpoint diagnostico non implica ripresa esatta completa: questa resta requisito fase5.': 'Congelare config, checkpoint diagnostico, log; priorità efficienza/recupero archi fase4. Ripresa esatta completa: requisito fase5, assente nel checkpoint diagnostico.',
    'Verificare configurazione candidata su storie validation escluse dall\'allenamento; confronto con inizializzazione e prompt fissi. Validation può guidare scelta; test finale resta chiuso.': 'Candidata: validation esclusa dal training; confronto inizializzazione, prompt fissi. Selezione su validation; test finale chiuso.',
    'Controllare generazione con cache, BOS/EOS e storie indipendenti; documentare capacità osservate e fallimenti senza promettere qualità da pretraining.': 'Verificare generazione: cache, BOS/EOS, storie indipendenti. Registrare capacità/fallimenti; qualità pretraining non presunta.',
    'Riaddestrare controlli quando serve confronto equo; rimozione a inferenza misura dipendenza, non sostituisce baseline allenata.': 'Confronto equo → riaddestrare controlli dove necessario. Rimozione a inferenza misura dipendenza; baseline allenata resta necessaria.',
    'Preferenza utente: includere sempre tabella aggiornata nel resoconto finale.': 'Resoconto finale: sempre tabella aggiornata.',
}
compressed = original
for old, new in replacements.items():
    assert old in compressed, old
    compressed = compressed.replace(old, new)
for pattern in [r'^#{1,6} .*$', r'`[^`]*`', r'\[[^\]]*\]\([^)]*\)', r'\d+(?:[.,]\d+)*', r'^\s*[-*+] ', r'^\|.*$']:
    assert re.findall(pattern, original, re.M) == re.findall(pattern, compressed, re.M), pattern
backup = p.with_name(p.stem + '.original.md')
assert not backup.exists(), backup
backup.write_bytes(p.read_bytes())
p.write_text(compressed, encoding='utf8')
print(json.dumps(dict(file=str(p),backup=str(backup),before_chars=len(original),after_chars=len(compressed),saved_pct=100*(1-len(compressed)/len(original)),preserved='headings, inline code, links, numbers, bullet structure, tables')))
