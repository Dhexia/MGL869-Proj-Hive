# Projet - ÉTS Montréal - Maîtrise - MGL869

Automne 2024

---

## Résumé
Ce dépôt contient le code du projet scolaire réalisé dans le cadre du cours MGL869 à l'ÉTS Montréal.  
L'objectif de ce laboratoire est d'implémenter une version simplifiée des algorithmes **régression logistique** et **forêt aléatoire** afin de prédire les bogues dans le logiciel [Hive](https://hive.com/).

---

## Cours
[MGL869-01 Sujets spéciaux I : génie logiciel (A2024)](https://www.etsmtl.ca/etudes/cours/mgl869-a24)

---

## Auteurs
- [William PHAN](mailto:william.phan.1@ens.etsmtl.ca)

---

## Superviseur
- [Mohammed SAYAGH, Ph.D., AP](mailto:mohammed.sayagh@etsmtl.ca)

---
## Instructions d'installation

Pour installer et exécuter ce projet, veuillez suivre les étapes ci-dessous :

1. **Créer un environnement virtuel :**  
   En raison de certaines bibliothèques qui ne sont pas pleinement compatibles avec les versions les plus récentes de Python, il est recommandé d'utiliser **Python 3.12**. Vous pouvez créer un environnement virtuel avec la commande suivante :  
   `python3.12 -m venv venv`  
   `source venv/bin/activate  # Pour activer l'environnement (Mac)`

2. **Installer les bibliothèques requises :**  
   Les dépendances nécessaires sont listées dans le fichier `requirements.txt`. Pour les installer, exécutez la commande suivante :  
   `pip install -r requirements.txt`

3. **Configuration des paramètres :**  
   - Les données et analyses ayant déjà été effectuées, elles sont disponibles dans les dossiers `data` à la racine ou `src/output`.
   - Certains paramètres dans le fichier `config.ini` sont définis sur `No` pour éviter de relancer les étapes déjà complétées.
   - Les paramètres d'entraînement, ainsi que ceux spécifiques à une machine Mac, sont également configurés dans `config.ini`.
   - Si vous utilisez une autre machine, vous devrez ajuster certains chemins, comme celui du logiciel **Understand SciTools** ou d'autres dépendances spécifiques.

---

### Source des données

Les données proviennent de la page web opensource Apache [issues Apache](https://issues.apache.org/jira/projects/HIVE/issues/HIVE-13282?filter=allopenissues).



