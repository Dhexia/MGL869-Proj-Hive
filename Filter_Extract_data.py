#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import git
from concurrent.futures import ThreadPoolExecutor
from typing import List
import asyncio
import aiofiles
import shutil
import tracemalloc

tracemalloc.start()
# In[2]:


folder_path = 'CSV exported'
# Lire tous les fichiers CSV du dossier
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# In[3]:


dataframes = [pd.read_csv(file, sep="^") for file in csv_files]
dataframes_filtered = []
for df in dataframes:
    fix_version_columns = [col for col in df.columns if col.startswith('Fix Version/s')]
    affects_version_columns = [col for col in df.columns if col.startswith('Affects Version/s')]

    df['Fix Versions Combined'] = df[fix_version_columns].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    df['Affects Versions Combined'] = df[affects_version_columns].apply(lambda x: ', '.join(x.dropna().astype(str)),
                                                                        axis=1)

    # Supprimer les colonnes originales
    df = df.drop(fix_version_columns, axis=1)
    df = df.drop(affects_version_columns, axis=1)

    keep: list = ['Issue key', 'Status', 'Resolution', 'Created', 'Fix Versions Combined', 'Affects Versions Combined']
    df = df.loc[:, keep]
    print(len(df.columns))

    dataframes_filtered.append(df)

# In[4]:


df_merged = pd.concat(dataframes_filtered, ignore_index=True, sort=False)

# In[5]:


print(df_merged.info)

# In[6]:


df_merged.to_csv('filtered_data.csv')

# ---
# GIT RESEARCH
# ---
# 
# 

# In[7]:


repo_path = r'C:\Users\moshi\Documents\projects\Informatique\ETS\MGL869\hive'
output_dir = 'commit_files'
os.makedirs(output_dir, exist_ok=True)

# In[8]:


repo = git.Repo(repo_path)

# In[9]:


# Fully load commits to avoid lazy-loading issues
commit_cache = {commit.hexsha: commit for commit in list(repo.iter_commits())}
# Ensure all commit messages are loaded
for commit in commit_cache.values():
    _ = commit.message


# In[10]:


# Function to find commits for a bug using the cache
def find_commits_for_bug(bug_id):
    return [commit for commit in commit_cache.values() if bug_id in commit.message]


# In[11]:


# Asynchronous function to process each bug
semaphore = asyncio.Semaphore(20)  # Adjust the limit as needed


async def process_bug(bug_id: str, output_dir: str):
    async with semaphore:
        commits = await asyncio.to_thread(find_commits_for_bug, bug_id)
        for commit in commits:
            filename = f"{bug_id}_{commit.hexsha}.txt"
            filepath = os.path.join(output_dir, filename)

            async with aiofiles.open(filepath, 'w') as file:
                await file.write(f"Bug ID: {bug_id}\n")
                await file.write(f"Commit: {commit.hexsha}\n")
                await file.write("Modified files:\n")
                for file_path in commit.stats.files:
                    await file.write(f"  {file_path}\n")


# In[12]:


# Main async function to run tasks concurrently
async def main():
    # Efficiently delete and recreate the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    tasks = []
    for i, bug_id in enumerate(df_merged['Issue key']):
        if i % 50 == 0:
            print(i)
        tasks.append(process_bug(bug_id, output_dir))

    # Run all tasks concurrently
    # Divide tasks into chunks of 100 to limit memory and CPU usage
    chunk_size = 100
    for i in range(0, len(tasks), chunk_size):
        await asyncio.gather(*tasks[i:i + chunk_size])


# Run the async main function
asyncio.run(main())
tracemalloc.stop()
