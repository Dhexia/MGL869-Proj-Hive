def update_commit_dataframe(commit_dataframe, jira_dataframe):
    """
    Merge commit_dataframe with jira_dataframe to replace the 'Commit' column 
    with the 'Affects Versions Combined' column from jira_dataframe.
    
    Parameters:
        commit_dataframe: DataFrame containing commit analysis.
        jira_dataframe: DataFrame containing JIRA issue details.
    
    Returns:
        DataFrame: Updated DataFrame with 'Version Affected' replacing 'Commit'.
    """

    merged_dataframe = commit_dataframe.merge(
        jira_dataframe[['Issue key', 'Affects Versions Combined']],
        on='Issue key',
        how='left'  # Conserver toutes les lignes de commit_dataframe
    )
    
    if 'Commit' in merged_dataframe.columns:
        merged_dataframe.drop(columns=['Commit'], inplace=True)
    merged_dataframe.rename(columns={'Affects Versions Combined': 'Version Affected'}, inplace=True)

    return merged_dataframe
