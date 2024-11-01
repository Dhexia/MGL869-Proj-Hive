# School Lab - ETS Montreal - Master's Degree - MGL869

Autumn 2024

---

## Summary
This repository contains the code for the school lab of the MGL869 course at ETS Montreal.
The goal of this lab is to implement simple version of algorithms "logistic regression" and "random forest" to predict bugs in software [Hive](https://hive.com/).
---

## Lesson
[MGL869-01 Sujets spéciaux I : génie logiciel (A2024)](https://www.etsmtl.ca/etudes/cours/mgl869-a24)

## Authors
- [Léo FORNOFF](leo.fornoff.1@ens.etsmtl.ca)
- [William PHAN](william.phan.1@ens.etsmtl.ca)
- [Yannis OUAKRIM](yannis.ouakrim.1@ens.etsmtl.ca)

## Supervisor

- [Mohammed SAYAGH, Ph.D., AP](mohammed.sayagh@etsmtl.ca)
---

## Part 1 - Data collection

### Data source

The data source is the opensource apache web page [issue apache](https://issues.apache.org/jira/projects/HIVE/issues/HIVE-13282?filter=allopenissues).

### Data filtering

For this lab, we will only use bugs that have impacted the **version 2.0.0** of the software that have already being solved.

We used this command to filter on the website:

```
project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND affectedVersion = 2.0.0
```

Then, export the data in CSV format with '^' as the separator.

> As we did not want to be hindered by the lack of data, the program works with a **CSV file with all fields**.

### Data cleaning
