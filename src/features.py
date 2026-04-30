def create_features(df):
    
    # 🔹 1. Tenure Grouping
    def tenure_group(x):
        if x <= 12:
            return "New"
        elif x <= 36:
            return "Mid"
        else:
            return "Loyal"
    
    df["Tenure Group"] = df["Tenure Months"].apply(tenure_group)

    # 🔹 2. Average Monthly Spend (behavior)
    df["Avg Monthly Spend"] = df["Total Charges"] / (df["Tenure Months"] + 1)

    # 🔹 3. Number of Services Used
    services = [
        'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]

    df["Total Services"] = df[services].apply(lambda row: sum(row == "Yes"), axis=1)

    return df