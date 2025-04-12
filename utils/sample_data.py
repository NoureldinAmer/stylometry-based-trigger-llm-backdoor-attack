def stratified_sample(df, n_samples=1000):
    """
    Sample n_samples from dataframe, preserving the distribution of user_id.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to sample from
    n_samples : int
        The number of samples to take

    Returns:
    --------
    pandas.DataFrame
        A sampled dataframe with n_samples rows
    """
    # Calculate the distribution of user_id in the original dataframe
    user_counts = df['user_id'].value_counts(normalize=True)

    # Initialize an empty dataframe to store the samples
    sampled_df = pd.DataFrame()

    # Calculate the number of samples to take from each user_id group
    # We need to handle rounding to ensure we get exactly n_samples rows
    samples_per_user = (user_counts * n_samples).astype(int)

    # If the sum is less than n_samples due to rounding,
    # add the missing samples to the most frequent users
    missing_samples = n_samples - samples_per_user.sum()
    if missing_samples > 0:
        # Get the users with the largest fractional parts
        fractional_parts = (user_counts * n_samples) - samples_per_user
        top_users = fractional_parts.sort_values(
            ascending=False).index[:missing_samples]
        for user in top_users:
            samples_per_user[user] += 1

    # Sample from each user_id group
    for user, n_samples_user in samples_per_user.items():
        if n_samples_user > 0:
            user_df = df[df['user_id'] == user]
            # If we need more samples than exist for this user, take all of them
            if n_samples_user >= len(user_df):
                user_samples = user_df
            else:
                # Otherwise take a random sample
                user_samples = user_df.sample(
                    n=n_samples_user, random_state=42)
            sampled_df = pd.concat([sampled_df, user_samples])

    return sampled_df


def stratified_sample(df, n_samples=1000):
    """
    Sample n_samples from dataframe, preserving the distribution of user_id.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to sample from
    n_samples : int
        The number of samples to take

    Returns:
    --------
    pandas.DataFrame
        A sampled dataframe with n_samples rows
    """
    # Calculate the distribution of user_id in the original dataframe
    user_counts = df['user_id'].value_counts(normalize=True)

    # Initialize an empty dataframe to store the samples
    sampled_df = pd.DataFrame()

    # Calculate the number of samples to take from each user_id group
    # We need to handle rounding to ensure we get exactly n_samples rows
    samples_per_user = (user_counts * n_samples).astype(int)

    # If the sum is less than n_samples due to rounding,
    # add the missing samples to the most frequent users
    missing_samples = n_samples - samples_per_user.sum()
    if missing_samples > 0:
        # Get the users with the largest fractional parts
        fractional_parts = (user_counts * n_samples) - samples_per_user
        top_users = fractional_parts.sort_values(
            ascending=False).index[:missing_samples]
        for user in top_users:
            samples_per_user[user] += 1

    # Sample from each user_id group
    for user, n_samples_user in samples_per_user.items():
        if n_samples_user > 0:
            user_df = df[df['user_id'] == user]
            # If we need more samples than exist for this user, take all of them
            if n_samples_user >= len(user_df):
                user_samples = user_df
            else:
                # Otherwise take a random sample
                user_samples = user_df.sample(
                    n=n_samples_user, random_state=42)
            sampled_df = pd.concat([sampled_df, user_samples])

    return sampled_df


# Example usage:
# sampled_df = stratified_sample(df, n_samples=1000)

# To verify the distribution is maintained:
def verify_distribution(original_df, sampled_df):
    """Compare original and sampled distributions of user_id"""
    original_dist = original_df['user_id'].value_counts(normalize=True)
    sampled_dist = sampled_df['user_id'].value_counts(normalize=True)

    comparison = pd.DataFrame({
        'Original %': original_dist * 100,
        'Sampled %': sampled_dist * 100,
        'Difference': (sampled_dist - original_dist) * 100
    }).round(2)

    return comparison
