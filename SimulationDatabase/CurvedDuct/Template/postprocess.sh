#!/bin/bash

# 1. Forza il SOLVER (es. simpleFoam) a rigenerare i campi di turulenza.
# Sostituisci "simpleFoam" con il nome del tuo solver se hai usato un comprimibile (es. rhoSimpleFoam)
simpleFoam -latestTime -postProcess -func "turbulenceFields(R,nuTilda)"

LATEST_TIME=$(ls -d [0-9]* 2>/dev/null | sort -n | tail -n 1)

# 2. Trova l'ultimo intervallo temporale numerico salvato
# Controllo di sicurezza: se non trova cartelle numeriche, si ferma
if [ -z "$LATEST_TIME" ]; then
    echo "Errore: Nessuna cartella temporale numerica trovata!"
    exit 1
fi

echo "Ultimo timestep rilevato: $LATEST_TIME"

# 3. Rinomina i file fisici usando la variabile appena estratta
mv "${LATEST_TIME}/turbulenceProperties:nuTilda" "${LATEST_TIME}/nuTilda"
mv "${LATEST_TIME}/turbulenceProperties:R" "${LATEST_TIME}/R"

# 4. Correggi l'header interno ai file con sed
sed -i "s/turbulenceProperties:nuTilda/nuTilda/g" "${LATEST_TIME}/nuTilda"
sed -i "s/turbulenceProperties:R/R/g" "${LATEST_TIME}/R"

# 5. Ora che i file fisici R e nuTilda sono stati scritti nella cartella 5000, calcola i gradienti
postProcess -latestTime -func "grad(nuTilda)"
postProcess -latestTime -func "grad(p)"
postProcess -latestTime -func "grad(U)"
postProcess -latestTime -func "grad(k)"
postProcess -latestTime -func "grad(nut)"
