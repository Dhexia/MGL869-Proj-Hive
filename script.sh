#!/bin/bash

export PATH=$PATH:/Applications/Understand.app/Contents/MacOS/

HIVE_DIR="../hive"
MGL_DIR="../MGL869-Lab-Hive"

# Charger les versions et leurs dates depuis versions.csv
while IFS=, read -r version release_date; do
    echo "Processing version $version with release date $release_date"
    cd "$HIVE_DIR" || exit

    # Trouver le dernier commit avant la date de sortie de la version
    commit=$(git rev-list -n 1 --before="$release_date" master)
    if [ -z "$commit" ]; then
        echo "No commit found before $release_date for version $version"
        continue
    fi

    echo "Analyzing version $version (commit $commit)"

	git stash --include-untracked
    git checkout "$commit" || exit

    und purge "$MGL_DIR/analysis.und" 

	und add "$HIVE_DIR" -db "$MGL_DIR/analysis.und"
    und analyze -db "$MGL_DIR/analysis.und"

	und report -db "$MGL_DIR/analysis.und" -metrics "AvgCyclomatic, CountLineCode, CountOutput, CountPath, MaxNesting" -format csv > "$MGL_DIR/full_metrics_$version.csv"
    # Filtrer le rapport pour n'inclure que les fichiers présents dans filtered_couples.csv
    awk -F, 'NR==FNR { files[$1]; next } $1 in files' "$MGL_DIR/filtered_couples.csv" "$MGL_DIR/full_metrics_$version.csv" > "$MGL_DIR/metrics_filtered_$version.csv"

    echo "Filtered metrics for version $version saved to metrics_filtered_$version.csv"
    git checkout master

done < "$MGL_DIR/versions.csv"
