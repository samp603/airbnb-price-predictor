import pandas as pd

def load_and_clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Remove price outliers
    df = df[df['price'] <= 1000]

    # Encode categorical columns
    df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == '__main__':
    load_and_clean_data('../data/AB_NYC_2019.csv', '../data/cleaned_data.csv')
